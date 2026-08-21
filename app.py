"""RescueNet ML API — load two lightweight multi-output models and predict food needs."""

from __future__ import annotations

import json
import logging
import os
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from food_schema import (
    CAT_BREEDS,
    DOG_BREEDS,
    EMERGENCY_MULTIPLIERS,
    EMERGENCY_TYPES,
    FEATURE_COLUMNS,
    HUMAN_FOOD,
    HUMAN_TARGETS,
    INTEGER_UNITS,
    PET_FOOD,
    PET_TARGETS,
    cooking_multiplier,
    normalize_pet_type,
    pet_size_load,
    round_amount,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

human_model = None
pet_model = None
label_encoder = None
food_info = {}
is_ready = False


def load_models():
    global human_model, pet_model, label_encoder, food_info, is_ready
    try:
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        human_path = os.path.join(MODEL_DIR, "human_food_model.pkl")
        pet_path = os.path.join(MODEL_DIR, "pet_food_model.pkl")
        info_path = os.path.join(MODEL_DIR, "food_info.json")

        if not (os.path.exists(encoder_path) and os.path.exists(human_path) and os.path.exists(pet_path)):
            logger.error("Model files missing in %s", MODEL_DIR)
            is_ready = False
            return

        label_encoder = joblib.load(encoder_path)
        human_model = joblib.load(human_path)
        pet_model = joblib.load(pet_path)
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                food_info = json.load(f)
        is_ready = True
        logger.info("Loaded human + pet multi-output models")
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        traceback.print_exc()
        is_ready = False


def count_people(person_details, fallback):
    children = int(fallback.get("children") or 0)
    elderly = int(fallback.get("elderly") or 0)
    pregnant = int(fallback.get("pregnant") or 0)
    vegetarian = 0
    diabetes = 0
    bp = 0
    heart = 0
    lactating = 0

    if isinstance(person_details, list) and person_details:
        children = sum(1 for p in person_details if int(p.get("age") or 0) < 12)
        elderly = sum(1 for p in person_details if int(p.get("age") or 0) > 60)
        pregnant = sum(
            1
            for p in person_details
            if p.get("isPregnant") is True or str(p.get("healthCondition") or "") == "Pregnant"
        )
        vegetarian = sum(
            1
            for p in person_details
            if str(p.get("dietaryRestriction") or "") in ("Vegetarian", "Vegan")
        )
        diabetes = sum(1 for p in person_details if str(p.get("healthCondition") or "") == "Diabetes")
        bp = sum(1 for p in person_details if str(p.get("healthCondition") or "") == "High Blood Pressure")
        heart = sum(1 for p in person_details if str(p.get("healthCondition") or "") == "Heart Condition")
        lactating = sum(1 for p in person_details if str(p.get("healthCondition") or "") == "Lactating")

    return {
        "children": children,
        "elderly": elderly,
        "pregnant": pregnant,
        "vegetarian_count": vegetarian,
        "diabetes_count": diabetes,
        "bp_count": bp,
        "heart_count": heart,
        "lactating_count": lactating,
    }


def summarize_pets(pets):
    if not isinstance(pets, list):
        pets = []
    dogs = [p for p in pets if normalize_pet_type(p.get("type")) == "dog"]
    cats = [p for p in pets if normalize_pet_type(p.get("type")) == "cat"]
    dog_weights = [float(p.get("weight") or 0) for p in dogs]
    cat_weights = [float(p.get("weight") or 0) for p in cats]
    dog_ages = [float(p.get("age") or 0) for p in dogs]
    cat_ages = [float(p.get("age") or 0) for p in cats]
    return {
        "pet_count": len(pets),
        "dog_count": len(dogs),
        "cat_count": len(cats),
        "total_dog_weight_kg": round(sum(dog_weights), 2),
        "total_cat_weight_kg": round(sum(cat_weights), 2),
        "avg_dog_age": round(sum(dog_ages) / len(dog_ages), 2) if dog_ages else 0.0,
        "avg_cat_age": round(sum(cat_ages) / len(cat_ages), 2) if cat_ages else 0.0,
        "dog_size_load": round(sum(pet_size_load(p) for p in dogs), 3),
        "cat_size_load": round(sum(pet_size_load(p) for p in cats), 3),
        "puppy_count": sum(1 for p in dogs if float(p.get("age") or 0) < 1),
        "kitten_count": sum(1 for p in cats if float(p.get("age") or 0) < 1),
        "senior_dog_count": sum(1 for p in dogs if float(p.get("age") or 0) >= 8),
        "senior_cat_count": sum(1 for p in cats if float(p.get("age") or 0) >= 10),
    }


def encode_emergency(name: str) -> int:
    try:
        return int(label_encoder.transform([name])[0])
    except Exception:
        return int(label_encoder.transform(["Flood"])[0])


def format_item(name, amount, unit, extra=None):
    payload = {
        "amount": amount,
        "unit": unit,
        "display": f"{amount} {unit}" if unit in INTEGER_UNITS else f"{amount:.2f} {unit}",
    }
    if extra:
        payload.update(extra)
    return payload


def rule_based(emergency_type, total_people, cooking_available, days, people_stats, pets):
    from food_schema import age_factor, breed_factor

    multiplier = EMERGENCY_MULTIPLIERS.get(emergency_type, 1.2)
    n = max(1, total_people)
    human = {}
    for food, spec in HUMAN_FOOD.items():
        amount = spec["base"] * total_people * days * multiplier
        amount *= cooking_multiplier(spec["needs_cooking"], cooking_available, spec["category"])
        if food == "Sugar":
            amount *= 1 - 0.75 * (people_stats["diabetes_count"] / n)
        if food == "Salt":
            amount *= 1 - 0.80 * ((people_stats["bp_count"] + people_stats["heart_count"]) / n)
        amount = round_amount(amount, spec["unit"])
        if amount:
            human[food] = format_item(food, amount, spec["unit"], {"cooking": spec["needs_cooking"]})

    pet_out = {}
    for food, spec in PET_FOOD.items():
        total = 0.0
        for pet in pets:
            pet_type = normalize_pet_type(pet.get("type"))
            if spec["for"] not in (pet_type, "both"):
                continue
            ref = 20.0 if pet_type == "dog" else 5.0
            weight = max(0.0, float(pet.get("weight") or 0))
            load = (weight / ref) * breed_factor(pet_type, pet.get("breed")) * age_factor(pet_type, pet.get("age"))
            total += spec["base"] * load * days * (0.92 + 0.08 * multiplier)
        amount = round_amount(total, spec["unit"])
        if amount:
            pet_out[food] = format_item(food, amount, spec["unit"], {"for": spec["for"]})
    return human, pet_out


load_models()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy" if is_ready else "loading",
            "is_ready": is_ready,
            "models_loaded": 2 if is_ready else 0,
            "human_targets": HUMAN_TARGETS,
            "pet_targets": PET_TARGETS,
        }
    )


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "running", "service": "RescueNet ML API", "version": "3.0"})


@app.route("/breeds", methods=["GET"])
def breeds():
    return jsonify({"dog_breeds": DOG_BREEDS, "cat_breeds": CAT_BREEDS, "emergency_types": EMERGENCY_TYPES})


def _predict_payload(data, use_ml: bool):
    emergency_type = data.get("emergency_type", "Flood")
    if emergency_type not in EMERGENCY_TYPES:
        emergency_type = "Flood"
    total_people = int(data.get("total_people") or 1)
    cooking_available = 1 if data.get("cooking_available", True) else 0
    days_required = int(data.get("days_required") or 3)
    person_details = data.get("person_details") or data.get("persons") or []
    pet_details = data.get("pet_details") or data.get("pets") or []

    people_stats = count_people(person_details, data)
    pet_stats = summarize_pets(pet_details)

    features = {
        "emergency_type_encoded": encode_emergency(emergency_type) if label_encoder is not None else 0,
        "total_people": total_people,
        "cooking_available": cooking_available,
        "children": people_stats["children"],
        "elderly": people_stats["elderly"],
        "pregnant": people_stats["pregnant"],
        "vegetarian_count": people_stats["vegetarian_count"],
        "diabetes_count": people_stats["diabetes_count"],
        "bp_count": people_stats["bp_count"],
        "heart_count": people_stats["heart_count"],
        "lactating_count": people_stats["lactating_count"],
        "days_required": days_required,
        **pet_stats,
    }
    feature_row = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLUMNS}])

    source = "ML Model (HistGradientBoosting)"
    if use_ml and is_ready:
        human_raw = human_model.predict(feature_row)[0]
        pet_raw = pet_model.predict(feature_row)[0]
        human = {}
        for name, raw in zip(HUMAN_TARGETS, human_raw):
            spec = HUMAN_FOOD[name]
            amount = round_amount(raw, spec["unit"])
            if amount:
                human[name] = format_item(name, amount, spec["unit"], {"cooking": spec["needs_cooking"]})
        pets = {}
        if pet_stats["pet_count"] > 0:
            for name, raw in zip(PET_TARGETS, pet_raw):
                spec = PET_FOOD[name]
                if spec["for"] == "dog" and pet_stats["dog_count"] == 0:
                    continue
                if spec["for"] == "cat" and pet_stats["cat_count"] == 0:
                    continue
                amount = round_amount(raw, spec["unit"])
                if amount:
                    pets[name] = format_item(name, amount, spec["unit"], {"for": spec["for"]})
    else:
        source = "Rule-based fallback"
        human, pets = rule_based(
            emergency_type,
            total_people,
            bool(cooking_available),
            days_required,
            people_stats,
            pet_details if isinstance(pet_details, list) else [],
        )

    water = human.pop("Water", None)
    if water is None:
        water = format_item("Water", round(total_people * 3.0 * days_required, 2), "L")

    return {
        "success": True,
        "predictions": {"human": human, "pets": pets, "water": water},
        "metadata": {
            "emergency_type": emergency_type,
            "total_people": total_people,
            "pet_count": pet_stats["pet_count"],
            "dog_count": pet_stats["dog_count"],
            "cat_count": pet_stats["cat_count"],
            "days_required": days_required,
            "cooking_available": bool(cooking_available),
            "source": source,
            "numpy_version": np.__version__,
        },
    }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        return jsonify(_predict_payload(data, use_ml=True))
    except Exception as exc:
        logger.error("Predict error: %s", exc)
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/fallback", methods=["POST"])
def fallback():
    try:
        data = request.get_json(force=True) or {}
        return jsonify(_predict_payload(data, use_ml=False))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/debug", methods=["GET"])
def debug():
    return jsonify(
        {
            "is_ready": is_ready,
            "feature_names": FEATURE_COLUMNS,
            "human_targets": HUMAN_TARGETS,
            "pet_targets": PET_TARGETS,
            "food_info_keys": list(food_info.keys()),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("Starting RescueNet ML API on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)

"""RescueNet ML API — predictions come only from loaded trained models."""

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
    EMERGENCY_TYPES,
    FEATURE_COLUMNS,
    HUMAN_FOOD,
    HUMAN_TARGETS,
    INTEGER_UNITS,
    PET_FOOD,
    PET_TARGETS,
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

        missing = [p for p in (encoder_path, human_path, pet_path) if not os.path.exists(p)]
        if missing:
            logger.error("Model files missing: %s", missing)
            is_ready = False
            return

        label_encoder = joblib.load(encoder_path)
        human_model = joblib.load(human_path)
        pet_model = joblib.load(pet_path)
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                food_info = json.load(f)
        is_ready = True
        logger.info("Loaded human + pet multi-output models from %s", MODEL_DIR)
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        traceback.print_exc()
        is_ready = False


def count_people(person_details, payload):
    children = int(payload.get("children") or 0)
    elderly = int(payload.get("elderly") or 0)
    pregnant = int(payload.get("pregnant") or 0)
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
    return int(label_encoder.transform([name])[0])


def format_item(amount, unit, extra=None):
    payload = {
        "amount": amount,
        "unit": unit,
        "display": f"{amount} {unit}" if unit in INTEGER_UNITS else f"{amount:.2f} {unit}",
    }
    if extra:
        payload.update(extra)
    return payload


load_models()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy" if is_ready else "models_not_loaded",
            "is_ready": is_ready,
            "models_loaded": 2 if is_ready else 0,
            "model_files": [
                "human_food_model.pkl",
                "pet_food_model.pkl",
                "label_encoder.pkl",
            ],
            "human_targets": HUMAN_TARGETS,
            "pet_targets": PET_TARGETS,
        }
    ), (200 if is_ready else 503)


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "status": "running" if is_ready else "models_not_loaded",
            "service": "RescueNet ML API",
            "version": "3.0",
            "is_ready": is_ready,
            "predict": "/predict",
        }
    )


@app.route("/breeds", methods=["GET"])
def breeds():
    return jsonify({"dog_breeds": DOG_BREEDS, "cat_breeds": CAT_BREEDS, "emergency_types": EMERGENCY_TYPES})


@app.route("/predict", methods=["POST"])
def predict():
    if not is_ready or human_model is None or pet_model is None or label_encoder is None:
        return jsonify(
            {
                "success": False,
                "error": "AI models are not loaded. Deploy human_food_model.pkl, pet_food_model.pkl, and label_encoder.pkl in models/.",
            }
        ), 503

    try:
        data = request.get_json(force=True) or {}
        emergency_type = data.get("emergency_type", "Flood")
        if emergency_type not in EMERGENCY_TYPES:
            return jsonify(
                {"success": False, "error": f"Unknown emergency_type. Use one of: {EMERGENCY_TYPES}"}
            ), 400

        total_people = int(data.get("total_people") or 1)
        cooking_available = 1 if data.get("cooking_available", True) else 0
        days_required = int(data.get("days_required") or 3)
        person_details = data.get("person_details") or data.get("persons") or []
        pet_details = data.get("pet_details") or data.get("pets") or []

        people_stats = count_people(person_details, data)
        pet_stats = summarize_pets(pet_details)

        try:
            encoded = encode_emergency(emergency_type)
        except Exception:
            return jsonify({"success": False, "error": "Label encoder cannot encode this emergency type. Retrain models."}), 400

        features = {
            "emergency_type_encoded": encoded,
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

        human_raw = human_model.predict(feature_row)[0]
        pet_raw = pet_model.predict(feature_row)[0]

        human = {}
        water = None
        for name, raw in zip(HUMAN_TARGETS, human_raw):
            spec = HUMAN_FOOD[name]
            amount = round_amount(raw, spec["unit"])
            item = format_item(amount, spec["unit"], {"cooking": spec["needs_cooking"]})
            if name == "Water":
                water = item
                continue
            if amount:
                human[name] = item

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
                    pets[name] = format_item(amount, spec["unit"], {"for": spec["for"]})

        if water is None:
            return jsonify({"success": False, "error": "Water target missing from the trained human model."}), 500

        return jsonify(
            {
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
                    "source": "ML Model (HistGradientBoosting)",
                    "numpy_version": np.__version__,
                    "models_used": ["human_food_model.pkl", "pet_food_model.pkl"],
                },
            }
        )
    except Exception as exc:
        logger.error("Predict error: %s", exc)
        traceback.print_exc()
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
    logger.info("Starting RescueNet ML API on port %s ready=%s", port, is_ready)
    app.run(host="0.0.0.0", port=port, debug=False)

"""
Generate 50,000 emergency food rows for RescueNet.
Run locally or in Google Colab after installing pinned numpy/pandas.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from food_schema import (
    CAT_BREED_PROFILE,
    CAT_BREEDS,
    DOG_BREED_PROFILE,
    DOG_BREEDS,
    EMERGENCY_MULTIPLIERS,
    EMERGENCY_TYPES,
    FEATURE_COLUMNS,
    HUMAN_FOOD,
    HUMAN_TARGETS,
    PET_FOOD,
    PET_TARGETS,
    age_factor,
    breed_factor,
    cooking_multiplier,
    normalize_pet_type,
    pet_size_load,
    round_amount,
)

N_ROWS = 50000
RANDOM_SEED = 42
CSV_NAME = "emergency_food_50000_v3.csv"


def generate_person(idx: int) -> dict:
    age = int(np.random.randint(1, 86))
    gender = random.choice(["Male", "Female", "Other"])
    dietary = random.choice(["None", "Vegetarian", "Vegan", "Gluten-Free", "Halal", "Kosher"])
    health = random.choice(
        ["None", "None", "None", "Diabetes", "High Blood Pressure", "Heart Condition", "Lactating"]
    )
    is_pregnant = bool(gender == "Female" and 18 <= age <= 45 and random.random() < 0.12)
    if is_pregnant:
        health = "Pregnant"
    return {
        "person_id": f"P{idx:04d}",
        "age": age,
        "gender": gender,
        "isPregnant": is_pregnant,
        "dietaryRestriction": dietary,
        "healthCondition": health,
    }


def generate_pet(idx: int, pet_type: str) -> dict:
    if pet_type == "dog":
        breed = random.choice(DOG_BREEDS)
        profile = DOG_BREED_PROFILE[breed]
        age = int(np.random.randint(0, 13))
        weight = round(float(np.random.uniform(profile["w_min"], profile["w_max"])), 1)
        dietary = random.choice(["Standard", "Senior", "Puppy/Kitten", "Weight Management", "Allergy-Prone"])
    else:
        breed = random.choice(CAT_BREEDS)
        profile = CAT_BREED_PROFILE[breed]
        age = int(np.random.randint(0, 16))
        weight = round(float(np.random.uniform(profile["w_min"], profile["w_max"])), 1)
        dietary = random.choice(["Standard", "Senior", "Puppy/Kitten", "Weight Management", "Allergy-Prone"])
    return {
        "pet_id": f"PET{idx:04d}",
        "type": pet_type.capitalize(),
        "breed": breed,
        "age": age,
        "weight": weight,
        "dietaryNeed": dietary,
    }


def summarize_people(people: list[dict]) -> dict:
    children = sum(1 for p in people if p["age"] < 12)
    elderly = sum(1 for p in people if p["age"] > 60)
    pregnant = sum(1 for p in people if p["isPregnant"] or p["healthCondition"] == "Pregnant")
    vegetarian = sum(1 for p in people if p["dietaryRestriction"] in ("Vegetarian", "Vegan"))
    diabetes = sum(1 for p in people if p["healthCondition"] == "Diabetes")
    bp = sum(1 for p in people if p["healthCondition"] == "High Blood Pressure")
    heart = sum(1 for p in people if p["healthCondition"] == "Heart Condition")
    lactating = sum(1 for p in people if p["healthCondition"] == "Lactating")
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


def summarize_pets(pets: list[dict]) -> dict:
    dogs = [p for p in pets if normalize_pet_type(p.get("type")) == "dog"]
    cats = [p for p in pets if normalize_pet_type(p.get("type")) == "cat"]
    dog_weights = [float(p["weight"]) for p in dogs]
    cat_weights = [float(p["weight"]) for p in cats]
    dog_ages = [float(p["age"]) for p in dogs]
    cat_ages = [float(p["age"]) for p in cats]
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
        "puppy_count": sum(1 for p in dogs if float(p["age"]) < 1),
        "kitten_count": sum(1 for p in cats if float(p["age"]) < 1),
        "senior_dog_count": sum(1 for p in dogs if float(p["age"]) >= 8),
        "senior_cat_count": sum(1 for p in cats if float(p["age"]) >= 10),
        "dog_breeds": "|".join(p["breed"] for p in dogs),
        "cat_breeds": "|".join(p["breed"] for p in cats),
    }


def human_food_amounts(total_people: int, days: int, cooking_available: bool, multiplier: float, stats: dict) -> dict:
    values = {}
    n = max(1, total_people)
    for food, spec in HUMAN_FOOD.items():
        amount = spec["base"] * total_people * days * multiplier
        amount *= cooking_multiplier(spec["needs_cooking"], cooking_available, spec["category"])

        if food == "Sugar":
            amount *= 1 - 0.75 * (stats["diabetes_count"] / n)
        if food == "Salt":
            amount *= 1 - 0.80 * ((stats["bp_count"] + stats["heart_count"]) / n)
        if food == "Cooking oil":
            amount *= 1 - 0.25 * (stats["heart_count"] / n)
        if food == "Milk powder":
            amount *= 1 + 0.55 * ((stats["pregnant"] + stats["lactating_count"] + stats["children"]) / n)
        if food == "Dhal (lentils)":
            amount *= 1 + 0.35 * (stats["vegetarian_count"] / n)
        if food == "Canned Tuna" or food == "Rice and chicken curry" or food == "Rice and canned fish":
            amount *= 1 - 0.85 * (stats["vegetarian_count"] / n)
        if food in ("Biscuits", "Milk powder", "Soup"):
            amount *= 1 + 0.25 * (stats["children"] / n)
        if food in ("Soup", "Rice"):
            amount *= 1 + 0.18 * (stats["elderly"] / n)

        amount *= float(np.random.uniform(0.97, 1.03))
        values[food] = round_amount(amount, spec["unit"])
    return values


def pet_food_amounts(pets: list[dict], days: int, multiplier: float) -> dict:
    values = {}
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
        total *= float(np.random.uniform(0.97, 1.03)) if total > 0 else 0.0
        values[food] = round_amount(total, spec["unit"])
    return values


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    encoder_map = {name: i for i, name in enumerate(EMERGENCY_TYPES)}
    rows = []
    pet_id = 0
    locations = [
        "Colombo", "Kandy", "Galle", "Jaffna", "Matara", "Negombo", "Anuradhapura",
        "Polonnaruwa", "Badulla", "Ratnapura", "Kurunegala", "Trincomalee", "Batticaloa",
    ]

    for i in range(n_rows):
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{n_rows}")

        emergency = random.choice(EMERGENCY_TYPES)
        multiplier = EMERGENCY_MULTIPLIERS[emergency]
        total_people = int(np.random.randint(1, 16))
        people = [generate_person(p + 1) for p in range(total_people)]
        stats = summarize_people(people)
        cooking_available = bool(random.random() < 0.55)
        days = int(np.random.randint(2, 8))

        pets = []
        if random.random() < 0.48:
            for _ in range(int(np.random.randint(1, 4))):
                pet_id += 1
                pets.append(generate_pet(pet_id, random.choice(["dog", "cat"])))
        pet_stats = summarize_pets(pets)

        human_vals = human_food_amounts(total_people, days, cooking_available, multiplier, stats)
        pet_vals = pet_food_amounts(pets, days, multiplier)

        timestamp = datetime(2024, 1, 1) + timedelta(
            days=int(np.random.randint(0, 366)),
            hours=int(np.random.randint(0, 24)),
        )

        row = {
            "emergency_type": emergency,
            "emergency_type_encoded": encoder_map[emergency],
            "total_people": total_people,
            "cooking_available": int(cooking_available),
            "days_required": days,
            "location": random.choice(locations),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            **stats,
            **{k: v for k, v in pet_stats.items()},
            **{f"human_{name}": human_vals[name] for name in HUMAN_TARGETS},
            **{f"pet_{name}": pet_vals[name] for name in PET_TARGETS},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.fillna(0)
    return df


def main():
    print("=" * 72)
    print("Generating RescueNet 50,000-row dataset")
    print("=" * 72)
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), CSV_NAME)
    df.to_csv(out_path, index=False)

    meta = {
        "rows": int(len(df)),
        "human_foods": HUMAN_TARGETS,
        "pet_foods": PET_TARGETS,
        "features": FEATURE_COLUMNS,
        "dog_breeds": DOG_BREEDS,
        "cat_breeds": CAT_BREEDS,
        "emergency_types": EMERGENCY_TYPES,
        "csv": CSV_NAME,
    }
    with open(os.path.join(os.path.dirname(__file__), "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(df)} rows -> {out_path}")
    print(f"Pet households: {(df['pet_count'] > 0).sum()}")
    print(f"Human food columns: {len(HUMAN_TARGETS)}")
    print(f"Pet food columns: {len(PET_TARGETS)}")


if __name__ == "__main__":
    main()

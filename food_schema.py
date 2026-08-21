"""
Shared RescueNet food / pet schema used by dataset generation, training, and the API.
Keep this file identical in Colab and on the server so predictions match training.
"""

INTEGER_UNITS = {"packets", "cans"}

HUMAN_FOOD = {
    "Rice": {"base": 0.4, "category": "staple", "unit": "kg", "needs_cooking": True},
    "Dhal (lentils)": {"base": 0.15, "category": "protein", "unit": "kg", "needs_cooking": True},
    "Cooking oil": {"base": 0.05, "category": "fat", "unit": "L", "needs_cooking": True},
    "Salt": {"base": 0.02, "category": "spice", "unit": "kg", "needs_cooking": True},
    "Sugar": {"base": 0.03, "category": "sweetener", "unit": "kg", "needs_cooking": True},
    "Biscuits": {"base": 0.5, "category": "snack", "unit": "packets", "needs_cooking": False},
    "Canned Tuna": {"base": 0.5, "category": "protein", "unit": "cans", "needs_cooking": False},
    "Water": {"base": 3.0, "category": "water", "unit": "L", "needs_cooking": False},
    "Milk powder": {"base": 0.05, "category": "dairy", "unit": "kg", "needs_cooking": False},
    "Potatoes": {"base": 0.2, "category": "vegetable", "unit": "kg", "needs_cooking": True},
    "Carrots": {"base": 0.1, "category": "vegetable", "unit": "kg", "needs_cooking": True},
    "Cabbage": {"base": 0.15, "category": "vegetable", "unit": "kg", "needs_cooking": True},
    "Pumpkin": {"base": 0.15, "category": "vegetable", "unit": "kg", "needs_cooking": True},
    "Brinjal": {"base": 0.1, "category": "vegetable", "unit": "kg", "needs_cooking": True},
    "Coconut": {"base": 0.2, "category": "fat", "unit": "kg", "needs_cooking": True},
    "Bread": {"base": 0.3, "category": "staple", "unit": "kg", "needs_cooking": False},
    "Noodles": {"base": 0.3, "category": "staple", "unit": "packets", "needs_cooking": False},
    "Sandwiches": {"base": 0.2, "category": "snack", "unit": "packets", "needs_cooking": False},
    "Boiled eggs": {"base": 0.2, "category": "protein", "unit": "packets", "needs_cooking": False},
    "Soup": {"base": 0.3, "category": "snack", "unit": "packets", "needs_cooking": False},
    "Rice and vegetable curry": {"base": 0.4, "category": "meal", "unit": "packets", "needs_cooking": False},
    "Rice and chicken curry": {"base": 0.4, "category": "meal", "unit": "packets", "needs_cooking": False},
    "Rice and canned fish": {"base": 0.4, "category": "meal", "unit": "packets", "needs_cooking": False},
    "Vegetable fried rice": {"base": 0.5, "category": "meal", "unit": "packets", "needs_cooking": False},
    "String hoppers with curry": {"base": 0.3, "category": "meal", "unit": "packets", "needs_cooking": False},
    "Roti with curry": {"base": 0.3, "category": "meal", "unit": "packets", "needs_cooking": False},
    "Paratha with curry": {"base": 0.3, "category": "meal", "unit": "packets", "needs_cooking": False},
}

PET_FOOD = {
    "Dog Food": {"base": 0.3, "for": "dog", "unit": "kg"},
    "Dog Treats": {"base": 0.05, "for": "dog", "unit": "packets"},
    "Dog Biscuits": {"base": 0.1, "for": "dog", "unit": "packets"},
    "Canned Dog Food": {"base": 0.4, "for": "dog", "unit": "cans"},
    "Cat Food": {"base": 0.15, "for": "cat", "unit": "kg"},
    "Cat Treats": {"base": 0.08, "for": "cat", "unit": "packets"},
    "Canned Cat Food": {"base": 0.2, "for": "cat", "unit": "cans"},
    "Cat Kibble": {"base": 0.12, "for": "cat", "unit": "kg"},
    "Pet Milk": {"base": 0.1, "for": "both", "unit": "L"},
    "Pet Supplements": {"base": 0.02, "for": "both", "unit": "packets"},
}

DOG_BREEDS = [
    "Labrador Retriever",
    "German Shepherd",
    "Golden Retriever",
    "Bulldog",
    "Poodle",
    "Beagle",
    "Rottweiler",
    "Dachshund",
    "Siberian Husky",
    "Great Dane",
]

CAT_BREEDS = [
    "Persian",
    "Siamese",
    "Maine Coon",
    "Ragdoll",
    "Bengal",
    "Sphynx",
    "British Shorthair",
    "Abyssinian",
    "Scottish Fold",
    "Oriental Shorthair",
]

DOG_BREED_PROFILE = {
    "Labrador Retriever": {"w_min": 25.0, "w_max": 36.0, "factor": 1.10},
    "German Shepherd": {"w_min": 22.0, "w_max": 40.0, "factor": 1.15},
    "Golden Retriever": {"w_min": 25.0, "w_max": 34.0, "factor": 1.10},
    "Bulldog": {"w_min": 18.0, "w_max": 25.0, "factor": 0.90},
    "Poodle": {"w_min": 6.0, "w_max": 20.0, "factor": 0.70},
    "Beagle": {"w_min": 9.0, "w_max": 11.0, "factor": 0.60},
    "Rottweiler": {"w_min": 35.0, "w_max": 50.0, "factor": 1.30},
    "Dachshund": {"w_min": 7.0, "w_max": 12.0, "factor": 0.45},
    "Siberian Husky": {"w_min": 16.0, "w_max": 27.0, "factor": 1.00},
    "Great Dane": {"w_min": 45.0, "w_max": 80.0, "factor": 1.60},
}

CAT_BREED_PROFILE = {
    "Persian": {"w_min": 3.5, "w_max": 5.5, "factor": 1.00},
    "Siamese": {"w_min": 2.5, "w_max": 5.0, "factor": 0.85},
    "Maine Coon": {"w_min": 5.0, "w_max": 8.5, "factor": 1.40},
    "Ragdoll": {"w_min": 4.5, "w_max": 7.5, "factor": 1.20},
    "Bengal": {"w_min": 3.5, "w_max": 6.5, "factor": 1.05},
    "Sphynx": {"w_min": 3.0, "w_max": 5.0, "factor": 0.80},
    "British Shorthair": {"w_min": 4.0, "w_max": 7.0, "factor": 1.10},
    "Abyssinian": {"w_min": 3.0, "w_max": 4.5, "factor": 0.85},
    "Scottish Fold": {"w_min": 3.5, "w_max": 6.0, "factor": 0.95},
    "Oriental Shorthair": {"w_min": 2.5, "w_max": 4.5, "factor": 0.80},
}

EMERGENCY_TYPES = [
    "Flood",
    "Earthquake",
    "Tsunami",
    "Cyclone",
    "Landslide",
    "Fire",
    "Drought",
    "Epidemic",
]

EMERGENCY_MULTIPLIERS = {
    "Flood": 1.20,
    "Earthquake": 1.30,
    "Tsunami": 1.40,
    "Cyclone": 1.25,
    "Landslide": 1.30,
    "Fire": 1.10,
    "Drought": 1.15,
    "Epidemic": 1.30,
}

FEATURE_COLUMNS = [
    "emergency_type_encoded",
    "total_people",
    "cooking_available",
    "children",
    "elderly",
    "pregnant",
    "vegetarian_count",
    "diabetes_count",
    "bp_count",
    "heart_count",
    "lactating_count",
    "days_required",
    "pet_count",
    "dog_count",
    "cat_count",
    "total_dog_weight_kg",
    "total_cat_weight_kg",
    "avg_dog_age",
    "avg_cat_age",
    "dog_size_load",
    "cat_size_load",
    "puppy_count",
    "kitten_count",
    "senior_dog_count",
    "senior_cat_count",
]

HUMAN_TARGETS = list(HUMAN_FOOD.keys())
PET_TARGETS = list(PET_FOOD.keys())


def normalize_pet_type(value):
    text = str(value or "dog").strip().lower()
    if text.startswith("cat"):
        return "cat"
    return "dog"


def breed_factor(pet_type, breed):
    breed = str(breed or "").strip()
    if pet_type == "dog":
        return DOG_BREED_PROFILE.get(breed, {}).get("factor", 1.0)
    return CAT_BREED_PROFILE.get(breed, {}).get("factor", 1.0)


def age_factor(pet_type, age):
    age = float(age or 0)
    if pet_type == "dog":
        if age < 1:
            return 1.30
        if age >= 8:
            return 0.85
        return 1.0
    if age < 1:
        return 1.35
    if age >= 10:
        return 0.85
    return 1.0


def pet_size_load(pet):
    pet_type = normalize_pet_type(pet.get("type"))
    weight = max(0.0, float(pet.get("weight") or 0))
    age = float(pet.get("age") or 0)
    return weight * breed_factor(pet_type, pet.get("breed")) * age_factor(pet_type, age)


def round_amount(amount, unit):
    amount = max(0.0, float(amount))
    if unit in INTEGER_UNITS:
        return int(round(amount))
    return round(amount, 2)


def cooking_multiplier(needs_cooking, cooking_available, category):
    if category == "water":
        return 1.0
    if cooking_available:
        if category == "meal":
            return 0.22
        if needs_cooking:
            return 1.0
        return 0.35
    if needs_cooking:
        return 0.12
    if category == "meal":
        return 1.45
    return 1.25

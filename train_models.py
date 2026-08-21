"""
Train lightweight HistGradientBoosting models for RescueNet.
Pin numpy==1.26.4 and scikit-learn==1.3.2 in Colab AND on Render so pickles load.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

from food_schema import (
    CAT_BREEDS,
    DOG_BREEDS,
    EMERGENCY_TYPES,
    FEATURE_COLUMNS,
    HUMAN_FOOD,
    HUMAN_TARGETS,
    PET_FOOD,
    PET_TARGETS,
)

warnings.filterwarnings("ignore")

CSV_NAME = "emergency_food_50000_v3.csv"
MODEL_DIR = "models"


def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray, names: list[str]) -> dict:
    results = {}
    r2_list = []
    mape_list = []
    for i, name in enumerate(names):
        yt = y_true.iloc[:, i].to_numpy(dtype=float)
        yp = y_pred[:, i]
        r2 = float(r2_score(yt, yp))
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mask = yt != 0
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if np.any(mask) else 0.0
        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}
        r2_list.append(r2)
        mape_list.append(mape)
        print(f"  {name:32s}  R2={r2:.4f}  MAPE={mape:5.1f}%  MAE={mae:.3f}")
    return {
        "per_target": results,
        "avg_r2": float(np.mean(r2_list)),
        "avg_mape": float(np.mean(mape_list)),
        "min_r2": float(np.min(r2_list)),
    }


def make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.08,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.05,
        early_stopping=False,
        random_state=42,
    )


def main():
    started = time.time()
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, CSV_NAME)
    model_dir = os.path.join(base, MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 72)
    print("RescueNet model training")
    print(f"  numpy={np.__version__}")
    import sklearn

    print(f"  sklearn={sklearn.__version__}")
    print("  model=HistGradientBoostingRegressor + MultiOutputRegressor")
    print("=" * 72)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run generate_dataset.py first.")

    df = pd.read_csv(csv_path)
    df = df.fillna(0).drop_duplicates()
    print(f"Loaded {len(df)} rows")

    le = LabelEncoder()
    le.fit(EMERGENCY_TYPES)
    df["emergency_type_encoded"] = le.transform(df["emergency_type"])

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y_human = df[[f"human_{name}" for name in HUMAN_TARGETS]]
    y_pet = df[[f"pet_{name}" for name in PET_TARGETS]]

    X_train, X_test, yh_train, yh_test, yp_train, yp_test = train_test_split(
        X, y_human, y_pet, test_size=0.2, random_state=42
    )
    print(f"Train={len(X_train)}  Test={len(X_test)}")

    print("\nTraining human food model...")
    human_model = MultiOutputRegressor(make_regressor(), n_jobs=-1)
    human_model.fit(X_train, yh_train)
    human_pred = human_model.predict(X_test)
    human_scores = evaluate(yh_test, human_pred, HUMAN_TARGETS)

    print("\nTraining pet food model...")
    pet_model = MultiOutputRegressor(make_regressor(), n_jobs=-1)
    pet_model.fit(X_train, yp_train)
    pet_pred = pet_model.predict(X_test)
    pet_scores = evaluate(yp_test, pet_pred, PET_TARGETS)

    joblib.dump(human_model, os.path.join(model_dir, "human_food_model.pkl"))
    joblib.dump(pet_model, os.path.join(model_dir, "pet_food_model.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

    with open(os.path.join(model_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(FEATURE_COLUMNS))

    food_info = {
        "human": {name: {"unit": spec["unit"], "needs_cooking": spec["needs_cooking"]} for name, spec in HUMAN_FOOD.items()},
        "pets": {name: {"unit": spec["unit"], "for": spec["for"]} for name, spec in PET_FOOD.items()},
        "dog_breeds": DOG_BREEDS,
        "cat_breeds": CAT_BREEDS,
        "emergency_types": EMERGENCY_TYPES,
    }
    with open(os.path.join(model_dir, "food_info.json"), "w", encoding="utf-8") as f:
        json.dump(food_info, f, indent=2)

    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "wrapper": "MultiOutputRegressor",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "features": FEATURE_COLUMNS,
        "human_food_targets": HUMAN_TARGETS,
        "pet_food_targets": PET_TARGETS,
        "human_avg_r2": human_scores["avg_r2"],
        "human_avg_mape": human_scores["avg_mape"],
        "human_min_r2": human_scores["min_r2"],
        "pet_avg_r2": pet_scores["avg_r2"],
        "pet_avg_mape": pet_scores["avg_mape"],
        "pet_min_r2": pet_scores["min_r2"],
        "data_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "human_results": human_scores["per_target"],
        "pet_results": pet_scores["per_target"],
        "notes": "Train with numpy==1.26.4 and scikit-learn==1.3.2. Copy models/ onto Render.",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - started
    print("\n" + "=" * 72)
    print(f"Human avg R2: {human_scores['avg_r2']:.4f}  min R2: {human_scores['min_r2']:.4f}")
    print(f"Pet    avg R2: {pet_scores['avg_r2']:.4f}  min R2: {pet_scores['min_r2']:.4f}")
    if human_scores["avg_r2"] >= 0.90 and pet_scores["avg_r2"] >= 0.90:
        print("Target accuracy 90%+ met.")
    else:
        print("Accuracy below 90%. Reduce dataset noise or increase max_iter.")
    print(f"Saved models in {model_dir}  ({elapsed/60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

base = Path(r"C:\src\rescue\rescue-ml-api")
schema = (base / "food_schema.py").read_text(encoding="utf-8")
gen = (base / "generate_dataset.py").read_text(encoding="utf-8")
train = (base / "train_models.py").read_text(encoding="utf-8")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass
    else:
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
    if not lines:
        lines = [""]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        md(
            "# RescueNet ML — Google Colab training\n\n"
            "Creates a **50,000-row** dataset and trains **two lightweight** models:\n\n"
            "- Human food (kg / L / whole packets)\n"
            "- Pet food for 10 dog breeds + 10 cat breeds (weight, breed, age, type)\n\n"
            "**Why these versions:** Render and this notebook must share `numpy==1.26.4` and "
            "`scikit-learn==1.3.2`. Training on Colab's default NumPy 2.x caused pickle load failures.\n\n"
            "**Model:** `HistGradientBoostingRegressor` inside `MultiOutputRegressor` — much smaller than "
            "one RandomForest pickle per food item, target **R² ≥ 0.90**.\n\n"
            "1. Run the install cell. If Colab asks, **Restart session**, then run install again.\n"
            "2. Run the remaining cells in order.\n"
            "3. Download `rescunet_models.zip` and put `models/` + `food_schema.py` in `rescue-ml-api`."
        ),
        code(
            "# STEP 1 — pin versions to match Render (avoids numpy._core pickle errors)\n"
            "!pip install -q numpy==1.26.4 pandas==2.1.4 scikit-learn==1.3.2 joblib==1.3.2\n"
            "import numpy, sklearn, pandas, joblib\n"
            "print('numpy', numpy.__version__)\n"
            "print('sklearn', sklearn.__version__)\n"
            "print('pandas', pandas.__version__)\n"
            "print('joblib', joblib.__version__)\n"
            "assert numpy.__version__.startswith('1.26'), 'Restart runtime after pip, then re-run this cell'\n"
            "assert sklearn.__version__.startswith('1.3.2'), 'Restart runtime after pip, then re-run this cell'"
        ),
        code("from pathlib import Path\nPath('food_schema.py').write_text(" + repr(schema) + ", encoding='utf-8')\nprint('Wrote food_schema.py')"),
        code("from pathlib import Path\nPath('generate_dataset.py').write_text(" + repr(gen) + ", encoding='utf-8')\nprint('Wrote generate_dataset.py')"),
        code("from pathlib import Path\nPath('train_models.py').write_text(" + repr(train) + ", encoding='utf-8')\nprint('Wrote train_models.py')"),
        code(
            "# STEP 5 — generate 50,000 rows (kg / L / whole packets)\n"
            "from generate_dataset import main as generate_main\n"
            "generate_main()\n"
            "import pandas as pd\n"
            "df = pd.read_csv('emergency_food_50000_v3.csv')\n"
            "print(df.shape)\n"
            "print('Human foods', [c for c in df.columns if c.startswith('human_')])\n"
            "print('Pet foods', [c for c in df.columns if c.startswith('pet_')])\n"
            "print(df[['total_people','cooking_available','days_required','pet_count','dog_size_load','human_Rice','human_Water','pet_Dog Food']].head())"
        ),
        code(
            "# STEP 6 — train two multi-output HistGradientBoosting models\n"
            "from train_models import main as train_main\n"
            "train_main()\n"
            "import json\n"
            "from pathlib import Path\n"
            "meta = json.loads(Path('models/metadata.json').read_text())\n"
            "print('Human avg R2', round(meta['human_avg_r2'], 4), 'min', round(meta['human_min_r2'], 4))\n"
            "print('Pet    avg R2', round(meta['pet_avg_r2'], 4), 'min', round(meta['pet_min_r2'], 4))"
        ),
        code(
            "# STEP 7 — zip models for download / Render deploy\n"
            "import shutil\n"
            "from google.colab import files\n"
            "shutil.make_archive('rescunet_models', 'zip', 'models')\n"
            "print('Copy models/ and food_schema.py into rescue-ml-api, then redeploy Render.')\n"
            "files.download('rescunet_models.zip')\n"
            "files.download('food_schema.py')\n"
            "files.download('emergency_food_50000_v3.csv')"
        ),
    ],
}

out = base / "RescueNet_Colab_Training.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out, "bytes", out.stat().st_size)

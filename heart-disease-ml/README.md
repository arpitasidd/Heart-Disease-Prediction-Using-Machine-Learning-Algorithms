# Heart Disease Prediction (VS Code Ready)

End‑to‑end ML project to predict cardiovascular disease risk using classic algorithms
(KNN, Random Forest, Decision Tree, and Logistic Regression). Includes a command‑line
interface (CLI) and a simple Streamlit app.

## Features
- **Auto data download** from UCI or OpenML (or load `data/heart.csv` if you put it there).
- Clean preprocessing with scikit‑learn Pipelines (missing values, scaling, encoding).
- Train/evaluate 4 models; save the best one automatically.
- Export metrics, ROC curve, and predictions to `outputs/` for further analysis (e.g., Tableau).
- One‑command terminal usage + scripts for macOS/Linux and Windows.
- Optional Streamlit dashboard for quick interactive exploration.

> Note: Your original bullets also mentioned merging sales/macroeconomic/weather data & Tableau dashboards.
> Those aren’t relevant to heart disease classification, so this repo focuses on the medical dataset only.
> We *do* export tidy CSVs for you to visualize in Tableau if you want.

## Quickstart

### 1) Create a virtual environment & install deps
```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train & evaluate (terminal / VS Code integrated terminal)
```bash
python app.py train --models knn rf dt lr --test-size 0.2
python app.py evaluate
```

### 3) Predict on a CSV (must include same columns as training set)
```bash
python app.py predict --input data/heart_sample.csv --output outputs/predictions.csv
```

### 4) Run the Streamlit app (optional)
```bash
streamlit run streamlit_app.py
```

### 5) Scripts (shortcut)
```bash
# macOS/Linux
bash scripts/run.sh

# Windows
scripts\run.bat
```

## Data Sources
- UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
- OpenML Heart Disease (fallback): https://www.openml.org/search?type=data&status=active&id=53

The loader tries: (1) local `data/heart.csv`, (2) UCI processed Cleveland dataset,
(3) OpenML dataset. If all fail (e.g., offline), it uses a small **synthetic** sample so
the pipeline still runs end‑to‑end. Replace `data/heart.csv` with your real data to train properly.

## Project Structure
```
heart-disease-ml/
├─ app.py
├─ streamlit_app.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ (optional) heart.csv
├─ models/
├─ outputs/
├─ scripts/
│  ├─ run.sh
│  └─ run.bat
└─ src/heart_disease/
   ├─ __init__.py
   ├─ data.py
   ├─ features.py
   ├─ models.py
   └─ evaluate.py
```

## Example Terminal Commands

Train all models and save best:
```bash
python app.py train --models knn rf dt lr
```

Evaluate saved best model:
```bash
python app.py evaluate
```

Predict:
```bash
python app.py predict --input data/heart_sample.csv --output outputs/predictions.csv
```

Enjoy! Open the folder in VS Code, create/activate your venv, and run the commands above.

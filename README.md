# ❤️ Heart Disease Prediction — Tabular ML (Streamlit + CLI)

End-to-end classification of **heart disease risk** from clinical features (UCI-style schema).  
Clean pipelines, pre-trained model, Streamlit dashboard, and CLI for train/evaluate/predict — all in a small, readable repo.

> ⚠️ Educational project. Not medical advice. Evaluate thoroughly and consult domain experts before real-world use.

---

## ✨ Features

* **Pre-trained model** (Logistic Regression) so you can run immediately.
* **Multiple algorithms** out of the box: KNN, Random Forest, Decision Tree, Logistic Regression.
* **Robust pipeline**: imputation, scaling, one-hot encoding, metrics & (optional) ROC plot.
* **Streamlit dashboard** for quick exploration and CSV predictions.
* **CLI** with simple commands: train / evaluate / predict.
* **Clean outputs**: metrics JSON, ROC image (if matplotlib installed), predictions CSV.

---

## 🗂 Project Structure

.
├─ app.py # CLI: train/evaluate/predict
├─ streamlit_app.py # Streamlit UI (CSV upload → predictions)
├─ requirements.txt # Full deps (incl. matplotlib)
├─ requirements_min_no_mpl.txt # Minimal deps (no matplotlib; safest on macOS 3.13)
├─ environment.yml # Conda env (Python 3.11 recommended on macOS)
├─ src/
│ └─ heart_disease/
│ ├─ data.py # Data loader (local → UCI/OpenML → synthetic fallback)
│ ├─ features.py # Preprocess pipeline builder
│ ├─ models.py # Model registry (knn/rf/dt/lr)
│ └─ evaluate.py # Metrics + (optional) ROC plotting
├─ models/
│ └─ best_model.joblib # Pre-trained model (ready to use)
├─ data/
│ └─ heart_sample.csv # Example input for predict
├─ outputs/
│ ├─ metrics_.json # Per-model metrics (after training/evaluate)
│ └─ roc_.png # ROC curves (if matplotlib available)
└─ scripts/
├─ run.sh # macOS/Linux quickstart script
└─ run.bat # Windows quickstart script

---

## ⚙️ Requirements

* **Python** 3.11+ (Conda env with 3.11 recommended on macOS)
* Packages:
  * Minimal: `pandas numpy scikit-learn joblib streamlit`
  * Optional (for ROC plots): `matplotlib`

---

## 🚀 Quick Start

```bash
# 1) Create & activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps (use minimal if you're on Python 3.13)
pip install --upgrade pip
pip install -r requirements_min_no_mpl.txt
# or
# pip install -r requirements.txt

# 3) Evaluate the pre-trained model (no training required)
python app.py evaluate

# 4) Run Streamlit dashboard
python -m streamlit run streamlit_app.py
Conda (recommended on macOS):
conda env create -f environment.yml
conda activate heart-ml
python app.py evaluate
python -m streamlit run streamlit_app.py
🧠 Strategy (default parameters)
Target: target (1 = disease, 0 = no disease)
Preprocessing: numeric median impute + scaling; categorical mode impute + one-hot
Models: knn, rf, dt, lr (choose any subset)
Split: stratified train/test (default test_size=0.2)
Selection: best model by ROC-AUC (if available)
You can change thresholds, models, and preprocessing in small, isolated blocks.
📤 Outputs (what to look at)
outputs/metrics_*.json: accuracy, precision, recall, F1, ROC-AUC (when available)
outputs/roc_*.png: ROC curves (if matplotlib installed)
outputs/predictions.csv: created via app.py predict for your uploads
🔄 Structural Flow (high-level)
flowchart TD
  A([Start]) --> B{Data Source}
  B -- local data/heart.csv --> C[Load CSV]
  B -- UCI/OpenML --> D[Download + Load]
  B -- none --> E[Synthetic Sample]

  C --> F[Preprocess Pipeline]
  D --> F
  E --> F

  F --> G{Mode}
  G -- Train --> H[Split & Fit (knn/rf/dt/lr)]
  H --> I[Evaluate + Metrics/ROC]
  I --> J[Save best_model.joblib]

  G -- Evaluate --> K[Load best model]
  K --> L[Compute metrics/ROC]

  G -- Predict --> M[Load best model]
  M --> N[Write predictions.csv]

  L --> O([End])
  J --> O
  N --> O
🧪 How we judge “working”
Consistent metrics (accuracy/F1) on holdout
Meaningful ROC-AUC (when class balance allows)
Stable preprocessing: same schema at predict-time
Reproducible runs with fixed random seeds for splits
🛠 Tweaks you can try (optional)
Add class weighting or threshold tuning for imbalanced sets.
Swap or add models (e.g., SVM, XGBoost), log best-of-N.
Feature selection or domain-derived features (e.g., risk buckets).
Cross-validation instead of single split (time/budget permitting).
🧰 Troubleshooting
Failed building wheel for matplotlib on macOS / Python 3.13
Use requirements_min_no_mpl.txt or Conda environment.yml (Python 3.11). ROC PNGs will be skipped, but metrics still save.
ModuleNotFoundError: <package>
Activate your venv/conda env and re-install:

source .venv/bin/activate
pip install -r requirements_min_no_mpl.txt
Predict schema mismatch
Ensure your CSV has the same feature columns used during training (the pipeline expects that schema).
📅 Roadmap
Add cross-validation & model ensembling
Model registry + YAML config
SHAP-based feature explanations in the UI
Dockerfile + CI for reproducible runs
📄 License
Choose a license (e.g., MIT) and add a LICENSE file.
🤝 Contributing
PRs welcome — keep changes focused and add a short note to the README.
👤 Author
Arpita Siddhabhatti

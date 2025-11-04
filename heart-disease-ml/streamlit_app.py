import streamlit as st
import pandas as pd
import joblib
from src.heart_disease.data import load_heart_dataframe
from src.heart_disease.features import build_preprocess

st.set_page_config(page_title="Heart Disease ML", layout="wide")

st.title("❤️ Heart Disease Prediction – Demo App")

with st.expander("Load Data & Preview", expanded=True):
    df = load_heart_dataframe(data_dir="data")
    st.write(df.head())
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

left, right = st.columns(2)
with left:
    st.subheader("Train Quick Model (Logistic Regression)")
    if st.button("Train Now"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        pre, *_ = build_preprocess(df)
        X = df.drop(columns=["target"])
        y = df["target"].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])
        pipe.fit(Xtr, ytr)
        acc = pipe.score(Xte, yte)
        st.success(f"Validation Accuracy: {acc:.3f}")
        joblib.dump(pipe, "models/streamlit_lr.joblib")
        st.info("Saved model -> models/streamlit_lr.joblib")

with right:
    st.subheader("Predict with Saved Model")
    model_choice = st.selectbox("Choose model file", ["models/best_model.joblib", "models/streamlit_lr.joblib"])
    uploaded = st.file_uploader("Upload CSV with same schema as training X", type=["csv"])
    if st.button("Run Prediction") and uploaded is not None:
        df_in = pd.read_csv(uploaded)
        pipe = joblib.load(model_choice)
        preds = pipe.predict(df_in)
        out = df_in.copy()
        out["prediction"] = preds
        st.write(out.head())
        st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

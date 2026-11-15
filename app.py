"""Streamlit app: accepts student details and displays the predicted academic performance category."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

ART = Path("artifacts")

st.set_page_config(page_title="Student Academic Performance Prediction", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    .hero {background: linear-gradient(120deg, #1e3a8a, #2563eb); color: white;
           padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: white; padding: 0;}
    .hero p {margin: .4rem 0 0 0; opacity: .9;}
    div[data-testid="stForm"] {border: 1px solid #e5e7eb; border-radius: 14px; padding: 1.5rem;
                               box-shadow: 0 2px 8px rgba(0,0,0,.05);}
    div[data-testid="stFormSubmitButton"] button {width: 100%; padding: .6rem; font-weight: 600;
                               background: #2563eb; color: white; border: none; border-radius: 8px;}
    div[data-testid="stFormSubmitButton"] button:hover {background: #1e40af; color: white;}
    .result {padding: 1.3rem 1.8rem; border-radius: 14px; margin-top: 1.2rem; border-left: 8px solid;}
    .result .small {margin: 0; font-size: .9rem; opacity: .8;}
    .result h2 {margin: .2rem 0 0 0; font-size: 1.8rem; padding: 0;}
    .Low     {background:#fef2f2; border-color:#dc2626; color:#7f1d1d;}
    .Average {background:#fffbeb; border-color:#d97706; color:#78350f;}
    .High    {background:#f0fdf4; border-color:#16a34a; color:#14532d;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Student Academic Performance Prediction</h1>
  <p>Enter the student details below to predict the academic performance category.</p>
</div>
""", unsafe_allow_html=True)

if not (ART / "meta.json").exists():
    st.error("Model not found. Run: python train.py --data student-mat.csv")
    st.stop()


@st.cache_resource
def load():
    meta = json.loads((ART / "meta.json").read_text())
    return meta, joblib.load(ART / "preprocessor.joblib"), keras.models.load_model(ART / "dnn_model.keras")


meta, pre, model = load()

LABELS = {
    "school": "School", "sex": "Sex", "age": "Age", "address": "Address (U = Urban, R = Rural)",
    "famsize": "Family size", "Pstatus": "Parents' status (T = Together, A = Apart)",
    "Medu": "Mother's education (0-4)", "Fedu": "Father's education (0-4)",
    "Mjob": "Mother's job", "Fjob": "Father's job", "reason": "Reason for choosing school",
    "guardian": "Guardian", "traveltime": "Travel time (1-4)", "studytime": "Weekly study time (1-4)",
    "failures": "Past class failures", "schoolsup": "Extra school support",
    "famsup": "Family educational support", "paid": "Paid extra classes",
    "activities": "Extra-curricular activities", "nursery": "Attended nursery school",
    "higher": "Wants higher education", "internet": "Internet access at home",
    "famrel": "Family relationship quality (1-5)",
    "freetime": "Free time after school (1-5)", "goout": "Going out with friends (1-5)",
    "Dalc": "Weekday alcohol consumption (1-5)", "Walc": "Weekend alcohol consumption (1-5)",
    "health": "Health status (1-5)", "absences": "Number of absences",
    "G1": "First period grade (0-20)", "G2": "Second period grade (0-20)",
}

GROUPS = {
    "👤 Demographic": ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
                       "Mjob", "Fjob", "guardian", "nursery"],
    "📚 Academic": ["G1", "G2", "failures", "schoolsup", "famsup", "paid", "higher",
                    "reason", "traveltime"],
    "🧑‍🤝‍🧑 Social": ["famrel", "freetime", "goout", "activities", "internet"],
    "🧠 Behavioural": ["studytime", "absences", "Dalc", "Walc", "health"],
}
features = meta["features"]
groups = {g: [n for n in names if n in features] for g, names in GROUPS.items()}
placed = {n for names in groups.values() for n in names}
leftover = [n for n in features if n not in placed]
if leftover:                                   # saved model is from an older training run
    st.error("Saved model is outdated (it uses: " + ", ".join(leftover) + "). "
             "Re-run: python train.py --data <path to student-mat.csv>")
    st.stop()
groups = {g: names for g, names in groups.items() if names}


def field(name, spec):
    label = LABELS.get(name, name)
    if spec["type"] == "cat":
        return st.selectbox(label, spec["options"],
                            format_func=lambda x: x.replace("_", " ").upper())
    if spec["min"] == spec["max"]:
        return spec["min"]
    return st.number_input(label, spec["min"], spec["max"], spec["default"], step=1)


values = {}
with st.form("student_form"):
    cols = st.columns(len(groups))
    for col, (title, names) in zip(cols, groups.items()):
        with col:
            st.markdown(f"#### {title}")
            for n in names:
                values[n] = field(n, features[n])
    submitted = st.form_submit_button("Predict")

if submitted:
    probs = model.predict(pre.transform(pd.DataFrame([values])), verbose=0)[0]
    category = meta["class_names"][int(probs.argmax())]
    st.markdown(f"""
    <div class="result {category}">
      <p class="small">Predicted Academic Performance Category</p>
      <h2>{category}</h2>
    </div>
    """, unsafe_allow_html=True)
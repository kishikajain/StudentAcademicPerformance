"""
DNN Project: Student Academic Performance Prediction (UCI Student Performance dataset)

Steps (as per project sheet): data preprocessing, EDA, feature selection,
DNN architecture design, model training, hyperparameter tuning,
performance evaluation, comparison with a traditional ML algorithm.

Usage:  python train.py --data student-mat.csv      (or student-por.csv)
"""
import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

CLASS_NAMES = ["Low", "Average", "High"]
OUT = Path("artifacts")


# 1. Data preprocessing -------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, sep=";")                      # UCI files use ';'
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing values:", int(df.isna().sum().sum()))
    df = df.drop_duplicates()
    # G3 (final grade, 0-20) -> performance category
    y = pd.cut(df["G3"], bins=[-1, 9, 14, 20], labels=[0, 1, 2]).astype(int).values
    X = df.drop(columns=["G3"])
    return df, X, y


def build_preprocessor(X):
    cat = X.select_dtypes("object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


# 2. Exploratory data analysis -----------------------------------------------
def run_eda(df):
    (OUT / "eda").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.histplot(df["G3"], bins=21, kde=True)
    plt.title("Distribution of final grade (G3)")
    plt.tight_layout(); plt.savefig(OUT / "eda" / "g3_distribution.png", dpi=120); plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout(); plt.savefig(OUT / "eda" / "correlation_heatmap.png", dpi=120); plt.close()
    print("EDA plots saved in artifacts/eda/")


# 3. Feature selection -------------------------------------------------------
# Input attributes grouped by type (demographic, academic, social, behavioural).
# 'romantic' is not used. The 5 most informative features of each group are kept.
GROUPS = {
    "Demographic": ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
                    "Mjob", "Fjob", "guardian", "nursery"],
    "Academic": ["G1", "G2", "failures", "schoolsup", "famsup", "paid", "higher",
                 "reason", "traveltime"],
    "Social": ["famrel", "freetime", "goout", "activities", "internet"],
    "Behavioural": ["studytime", "absences", "Dalc", "Walc", "health"],
}


def select_features(X, y, per_group=5):
    cat = X.select_dtypes("object").columns.tolist()
    Xo = X.copy()
    Xo[cat] = OrdinalEncoder().fit_transform(Xo[cat])
    mi = pd.Series(mutual_info_classif(Xo, y, discrete_features=[c in cat for c in X.columns],
                                       random_state=SEED), index=X.columns)
    selected = []
    for group, cols in GROUPS.items():
        top = mi[cols].sort_values(ascending=False).head(per_group).index.tolist()
        print(f"{group}: {top}")
        selected += top
    return selected


# 4. DNN architecture ---------------------------------------------------------
def build_dnn(input_dim, units=64, n_layers=3, dropout=0.3, lr=1e-3):
    model = keras.Sequential([keras.Input(shape=(input_dim,))])
    for i in range(n_layers):
        model.add(layers.Dense(max(units // 2 ** i, 8), activation="relu"))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(model, X, y, batch=32, verbose=0):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    return model.fit(X, y, validation_split=0.2, epochs=200, batch_size=batch,
                     callbacks=[stop], verbose=verbose)


# 5. Hyperparameter tuning (random search) ---------------------------------
def tune(X, y, trials=10):
    space = {"units": [32, 64, 128], "n_layers": [2, 3, 4],
             "dropout": [0.2, 0.3, 0.5], "lr": [1e-2, 1e-3, 5e-4], "batch": [16, 32, 64]}
    rng = random.Random(SEED)
    best, best_acc = None, -1
    for t in range(trials):
        p = {k: rng.choice(v) for k, v in space.items()}
        keras.backend.clear_session()
        m = build_dnn(X.shape[1], p["units"], p["n_layers"], p["dropout"], p["lr"])
        h = train(m, X, y, p["batch"])
        acc = max(h.history["val_accuracy"])
        print(f"  trial {t+1}/{trials} {p} -> val acc {acc:.3f}")
        if acc > best_acc:
            best, best_acc = p, acc
    print("Best hyperparameters:", best)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df, X, y = load_data(args.data)
    run_eda(df)
    selected = select_features(X, y)
    X = X[selected]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = build_preprocessor(X)
    Xtr_t = pre.fit_transform(Xtr)       # fit on training data only
    Xte_t = pre.transform(Xte)

    print("\nTuning hyperparameters...")
    best = tune(Xtr_t, ytr)

    print("\nTraining final DNN...")
    keras.backend.clear_session()
    dnn = build_dnn(Xtr_t.shape[1], best["units"], best["n_layers"], best["dropout"], best["lr"])
    dnn.summary()
    train(dnn, Xtr_t, ytr, best["batch"])

    # 6. Evaluation + comparison with a traditional ML algorithm
    dnn_pred = dnn.predict(Xte_t, verbose=0).argmax(1)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(Xtr_t, ytr)
    rf_pred = rf.predict(Xte_t)

    print("\n=== DNN performance ===")
    print(classification_report(yte, dnn_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(yte, dnn_pred))
    print("\n=== Comparison (test accuracy) ===")
    print(f"DNN           : {accuracy_score(yte, dnn_pred):.4f}")
    print(f"Random Forest : {accuracy_score(yte, rf_pred):.4f}")

    # save for the Streamlit app
    dnn.save(OUT / "dnn_model.keras")
    joblib.dump(pre, OUT / "preprocessor.joblib")
    feats = {}
    for c in selected:
        if X[c].dtype == "object":
            feats[c] = {"type": "cat", "options": sorted(X[c].unique().tolist())}
        else:
            feats[c] = {"type": "num", "min": int(X[c].min()), "max": int(X[c].max()),
                        "default": int(X[c].median())}
    (OUT / "meta.json").write_text(json.dumps({"class_names": CLASS_NAMES, "features": feats}, indent=2))
    print("\nSaved model to artifacts/. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
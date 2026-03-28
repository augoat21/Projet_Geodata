"""
Entraînement du modèle XGBoost pour la prédiction de risque de feu.

Usage :
    python -m ml.train_model --dataset ml/datasets/dataset_France_Portugal_Spain_2020-2024.csv
"""

import argparse
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Features utilisées par le modèle
FEATURE_COLS = [
    "latitude",
    "longitude",
    "temp_max",
    "humidity_min",
    "wind_max",
    "precip_sum",
    "month",
    "day_of_year",
    "month_sin",
    "month_cos",
]


def load_dataset(path):
    """Charge et prépare le dataset."""
    df = pd.read_csv(path)
    print(f"Dataset chargé : {len(df)} lignes")
    print(f"  Positifs (feu) : {(df['label'] == 1).sum()}")
    print(f"  Négatifs (pas de feu) : {(df['label'] == 0).sum()}")

    # Vérifier les colonnes
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ATTENTION: colonnes manquantes : {missing}")
        # Remplir precip_sum si absent
        if "precip_sum" in missing:
            df["precip_sum"] = 0
            missing.remove("precip_sum")

    # Remplacer les NaN dans precip_sum par 0 (pas de pluie)
    df["precip_sum"] = df["precip_sum"].fillna(0)

    # Supprimer les lignes avec NaN dans les features critiques
    df.dropna(subset=[c for c in FEATURE_COLS if c != "precip_sum"], inplace=True)
    print(f"  Après nettoyage : {len(df)} lignes")

    return df


def train(df, test_size=0.2):
    """Entraîne le modèle XGBoost et évalue ses performances."""
    X = df[FEATURE_COLS].copy()
    y = df["label"].copy()

    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # Configuration XGBoost
    model = XGBClassifier(
        n_estimators=300,         # nombre d'arbres
        max_depth=6,              # profondeur max de chaque arbre
        learning_rate=0.1,        # pas d'apprentissage (shrinkage)
        subsample=0.8,            # % de lignes utilisées par arbre
        colsample_bytree=0.8,    # % de features utilisées par arbre
        min_child_weight=5,       # régularisation (min samples par feuille)
        gamma=0.1,                # régularisation (min gain pour split)
        reg_alpha=0.1,            # régularisation L1
        reg_lambda=1.0,           # régularisation L2
        scale_pos_weight=1,       # équilibre des classes
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # Entraînement avec early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    print("\n" + "=" * 60)
    print("RÉSULTATS")
    print("=" * 60)

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=["Pas de feu", "Feu"]))

    print("Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Vrais négatifs  : {cm[0][0]}")
    print(f"  Faux positifs   : {cm[0][1]}")
    print(f"  Faux négatifs   : {cm[1][0]}")
    print(f"  Vrais positifs  : {cm[1][1]}")

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nAUC-ROC : {auc:.4f}")

    # Cross-validation 5-fold
    print("\nCross-validation 5-fold :")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  AUC moyen : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Importance des features
    print("\nImportance des features :")
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {feat:20s} : {imp:.4f} {bar}")

    return model, {
        "auc_roc": float(auc),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURE_COLS,
    }


def save_model(model, metrics, name="fire_risk_xgboost"):
    """Sauvegarde le modèle et ses métriques."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{name}.joblib"
    metrics_path = MODEL_DIR / f"{name}_metrics.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nModèle sauvegardé : {model_path}")
    print(f"Métriques sauvegardées : {metrics_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement XGBoost pour prédiction de feux")
    parser.add_argument("--dataset", required=True, help="Chemin vers le dataset CSV")
    parser.add_argument("--name", default="fire_risk_xgboost", help="Nom du modèle")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    model, metrics = train(df)
    save_model(model, metrics, args.name)

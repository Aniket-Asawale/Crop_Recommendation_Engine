"""
Baseline + Ensemble Model Training for Crop Recommendation Engine.

Trains 6 baseline models + voting ensemble, evaluates on temporal holdout,
applies probability calibration (temperature scaling), saves best model.

Usage: python Crop_Recommendation_Engine/models/baseline_models.py

Reads:  data/processed/features.csv
        data/synthetic/crop_recommendation_dataset.csv (for season_year split)
Writes: models/model_registry/best_model_YYYY_MM.pkl
        models/model_registry/calibrator_YYYY_MM.pkl
        models/model_registry/scaler.pkl
        models/model_registry/training_log.csv
        data/processed/model_comparison_report.txt
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path for central imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── Central config + feature engineering ───
from config import (
    BASE_DIR, FEATURES_CSV, RAW_CSV, ENCODERS_JSON,
    REGISTRY_DIR, REPORT_PATH,
    CONFIDENCE_HIGH, CONFIDENCE_UNCERTAIN,
    RANDOM_STATE, LABEL_NOISE_RATE, SMOTE_K_NEIGHBORS,
    TEMPORAL_TRAIN_YEARS, TEMPORAL_TEST_YEAR, MIN_CLASS_SAMPLES,
    MODEL_STAMP,
)
from feature_engineering import META_COLS, TARGET_COL, add_interaction_features

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _inject_label_noise(y: np.ndarray, noise_rate: float = LABEL_NOISE_RATE,
                        rng_seed: int = RANDOM_STATE) -> np.ndarray:
    """Randomly flip a fraction of training labels to neighbouring classes.

    This prevents the model from learning to be 100% confident on every
    sample — it will encounter contradictory examples near decision boundaries,
    teaching it to output calibrated (non-extreme) probabilities.
    """
    rng = np.random.RandomState(rng_seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    classes = np.unique(y)
    for idx in flip_idx:
        # Pick a random *different* class
        choices = classes[classes != y[idx]]
        y_noisy[idx] = rng.choice(choices)
    return y_noisy


class TemperatureScaler:
    """Post-hoc probability calibration via temperature scaling.

    Learns a single scalar T on a calibration set that softens
    (or sharpens) the predicted probabilities:
        calibrated_prob = softmax(logits / T)

    For tree models that output probabilities directly, we treat
    log(prob) as logits.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """Find optimal temperature on calibration data."""
        # Clamp to avoid log(0)
        probs_safe = np.clip(probs, 1e-10, 1.0)
        logits = np.log(probs_safe)

        def nll(T):
            scaled = logits / T
            # softmax
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            softmax_s = exp_s / exp_s.sum(axis=1, keepdims=True)
            # negative log-likelihood
            return -np.mean(np.log(softmax_s[np.arange(len(y_true)), y_true] + 1e-10))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature to new probabilities."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        logits = np.log(probs_safe)
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)



def load_data():
    """Load features and split by temporal holdout (2023-2024 train, 2025 test)."""
    df = pd.read_csv(FEATURES_CSV)

    # Drop tiny classes (< 5 samples) — can't stratify or learn them
    class_counts = df["crop_label"].value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if rare_classes:
        logger.warning("Dropping rare classes (%d): %s", len(rare_classes), rare_classes)
        df = df[~df["crop_label"].isin(rare_classes)].reset_index(drop=True)
        le = LabelEncoder()
        df[TARGET_COL] = le.fit_transform(df["crop_label"])

    # Add interaction features
    df = add_interaction_features(df)

    feature_cols = [c for c in df.columns if c not in META_COLS and c != TARGET_COL]

    # Temporal holdout: train on 2023-2024, test on 2025
    train_mask = df["season_year"].isin(TEMPORAL_TRAIN_YEARS)
    test_mask = df["season_year"] == TEMPORAL_TEST_YEAR

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, TARGET_COL].values

    # Apply SMOTE to training data only
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    logger.info("SMOTE: %d training rows after oversampling", len(X_train))

    # Inject 5% label noise to prevent overconfidence
    y_train = _inject_label_noise(y_train, noise_rate=LABEL_NOISE_RATE)
    logger.info("Label noise: 5%% injected (%d flipped labels)", int(len(y_train) * 0.05))

    # Crop label names for the report
    labels = df["crop_label"].unique()
    labels.sort()

    logger.info("Features: %d | Classes: %d", len(feature_cols), len(labels))
    logger.info("Train: %d rows (SMOTE+noise) | Test: %d rows (2025)", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, feature_cols, labels


def build_models(X_train_scaled, y_train):
    """Build all 6 baseline + 2 ensemble models."""
    # Import boosting libs
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    n_classes = len(np.unique(y_train))

    # ── Phase 1: Baselines (tuned hyperparams) ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=3,
        min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.7, colsample_bylevel=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        num_class=n_classes, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=12, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.7,
        num_leaves=63, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )
    svm = SVC(
        C=50, kernel="rbf", gamma="scale",
        decision_function_shape="ovr", probability=True,
        random_state=RANDOM_STATE,
    )
    knn = KNeighborsClassifier(
        n_neighbors=5, weights="distance", metric="minkowski", p=1,
        n_jobs=-1,
    )
    nb = GaussianNB()

    # ── Phase 2: Ensembles ──
    # Voting: soft voting over top 3 tree models
    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft", n_jobs=-1,
    )
    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "SVM (RBF)": svm,
        "KNN": knn,
        "Voting (RF+XGB+LGBM)": voting,
    }
    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test,
                       X_train_scaled, X_test_scaled, feature_cols, labels):
    """Train all models, evaluate, return results sorted by accuracy."""
    results = []
    trained = {}

    # Models that need scaling
    needs_scaling = {"SVM (RBF)", "KNN", "Naive Bayes"}

    for name, model in models.items():
        logger.info("Training %s...", name)
        t0 = time.time()

        X_tr = X_train_scaled if name in needs_scaling else X_train
        X_te = X_test_scaled if name in needs_scaling else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        elapsed = time.time() - t0

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt,
            "train_time_s": round(elapsed, 1),
        })
        trained[name] = model

        logger.info("%s: Acc=%.4f  F1m=%.4f  F1w=%.4f  (%.1fs)", name, acc, f1_mac, f1_wt, elapsed)

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results, trained


def save_best_model(results, trained, scaler, X_test, X_test_scaled,
                    y_test, feature_cols, labels):
    """Save best model, scaler, calibrator, and training log."""
    best = results[0]
    best_name = best["model"]
    best_model = trained[best_name]
    needs_scaling = {"SVM (RBF)", "KNN", "Naive Bayes"}
    stamp = datetime.now().strftime("%Y_%m")

    # Save model
    model_path = REGISTRY_DIR / f"best_model_{stamp}.pkl"
    joblib.dump(best_model, model_path)

    # Save scaler
    scaler_path = REGISTRY_DIR / f"scaler_{stamp}.pkl"
    joblib.dump(scaler, scaler_path)

    # ── Temperature scaling calibration ──
    X_te = X_test_scaled if best_name in needs_scaling else X_test
    raw_probs = best_model.predict_proba(X_te)
    calibrator = TemperatureScaler().fit(raw_probs, y_test)
    cal_probs = calibrator.calibrate(raw_probs)

    calibrator_path = REGISTRY_DIR / f"calibrator_{stamp}.pkl"
    # Save as dict to avoid pickle issues with class definitions
    joblib.dump({"temperature": calibrator.temperature}, calibrator_path)
    logger.info("Temperature scaling: T=%.3f", calibrator.temperature)

    # ── Calibration metrics ──
    y_pred_raw = best_model.predict(X_te)
    y_pred_cal = cal_probs.argmax(axis=1)
    raw_max = raw_probs.max(axis=1)
    cal_max = cal_probs.max(axis=1)

    # Brier score (per-class average)
    classes = best_model.classes_
    y_bin = label_binarize(y_test, classes=classes)
    brier_raw = np.mean([brier_score_loss(y_bin[:, i], raw_probs[:, i])
                         for i in range(len(classes))])
    brier_cal = np.mean([brier_score_loss(y_bin[:, i], cal_probs[:, i])
                         for i in range(len(classes))])

    cal_stats = {
        "temperature": calibrator.temperature,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "mean_prob_raw": float(raw_max.mean()),
        "mean_prob_cal": float(cal_max.mean()),
        "pct_over99_raw": float((raw_max > 0.99).mean()),
        "pct_over99_cal": float((cal_max > 0.99).mean()),
        "pct_uncertain_raw": float((raw_max < CONFIDENCE_UNCERTAIN).mean()),
        "pct_uncertain_cal": float((cal_max < CONFIDENCE_UNCERTAIN).mean()),
    }

    # Save training log CSV
    log_path = REGISTRY_DIR / "training_log.csv"
    log_df = pd.DataFrame(results)
    log_df["timestamp"] = datetime.now().isoformat()
    log_df["dataset_version"] = "v1.1 — 7200 rows + 5% label noise"
    log_df["train_period"] = "2023-2024"
    log_df["test_period"] = "2025"
    log_df["temperature"] = calibrator.temperature
    log_df.to_csv(log_path, index=False)

    # Classification report for best model
    y_pred = best_model.predict(X_te)
    report = classification_report(y_test, y_pred, target_names=labels,
                                   zero_division=0)

    logger.info("Best model: %s (Acc=%.4f)", best_name, best["accuracy"])
    logger.info("Saved to: %s", model_path)
    logger.info("Calibrator: %s", calibrator_path)
    return best, report, cal_stats, calibrator, raw_probs, cal_probs


def generate_report(results, best, class_report, feature_cols, labels,
                    cal_stats=None):
    """Generate comparison report with calibration metrics."""
    lines = [
        "=" * 70,
        "MODEL COMPARISON REPORT — Crop Recommendation Engine",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Features: {len(feature_cols)} | Classes: {len(labels)}",
        f"Split: Temporal holdout (Train=2023-2024, Test=2025)",
        f"Training noise: 5% label noise injected for calibration",
        "",
        "── Model Rankings (by Accuracy) ──",
        f"{'Rank':<5} {'Model':<30} {'Accuracy':>10} {'F1-macro':>10} {'F1-weighted':>12} {'Time(s)':>8}",
        "-" * 75,
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i:<5} {r['model']:<30} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
            f"{r['f1_weighted']:>12.4f} {r['train_time_s']:>8.1f}"
        )

    lines.extend([
        "",
        f"── Best Model: {best['model']} ──",
        f"Accuracy:    {best['accuracy']:.4f}",
        f"F1-macro:    {best['f1_macro']:.4f}",
        f"F1-weighted: {best['f1_weighted']:.4f}",
        "",
        "── Per-Class Classification Report (Best Model) ──",
        class_report,
    ])

    # Calibration section
    if cal_stats:
        lines.extend([
            "",
            "── Probability Calibration (Temperature Scaling) ──",
            f"  Temperature:           {cal_stats['temperature']:.3f}",
            f"  Brier score (raw):     {cal_stats['brier_raw']:.5f}",
            f"  Brier score (cal):     {cal_stats['brier_calibrated']:.5f}",
            f"  Mean P(max) raw:       {cal_stats['mean_prob_raw']:.4f}",
            f"  Mean P(max) cal:       {cal_stats['mean_prob_cal']:.4f}",
            f"  % over 0.99 (raw):     {cal_stats['pct_over99_raw']*100:.1f}%",
            f"  % over 0.99 (cal):     {cal_stats['pct_over99_cal']*100:.1f}%",
            f"  % uncertain (raw):     {cal_stats['pct_uncertain_raw']*100:.1f}% (<{CONFIDENCE_UNCERTAIN})",
            f"  % uncertain (cal):     {cal_stats['pct_uncertain_cal']*100:.1f}% (<{CONFIDENCE_UNCERTAIN})",
            "",
            "── Confidence Thresholds ──",
            f"  HIGH confidence:       >= {CONFIDENCE_HIGH}",
            f"  UNCERTAIN threshold:   < {CONFIDENCE_UNCERTAIN}",
            f"  Crops below UNCERTAIN threshold are flagged for manual review.",
        ])

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=" * 70)
    logger.info("MODEL TRAINING — Crop Recommendation Engine")
    logger.info("=" * 70)

    # Load & split
    logger.info("[1] Loading data...")
    X_train, X_test, y_train, y_test, feature_cols, labels = load_data()

    # Scale (for SVM, KNN, NB)
    logger.info("[2] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Scaler fitted on %d training rows", X_train_scaled.shape[0])

    # Build models
    logger.info("[3] Building models...")
    models = build_models(X_train_scaled, y_train)
    logger.info("Built %d models", len(models))

    # Train & evaluate
    logger.info("[4] Training & evaluating...")
    results, trained = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, feature_cols, labels,
    )

    # Save best + calibrate
    logger.info("[5] Saving best model & calibrating probabilities...")
    best, class_report, cal_stats, calibrator, raw_probs, cal_probs = \
        save_best_model(
            results, trained, scaler, X_test, X_test_scaled,
            y_test, feature_cols, labels,
        )

    # Report
    logger.info("[6] Generating report...")
    report = generate_report(results, best, class_report, feature_cols, labels,
                             cal_stats)
    # Print with ascii fallback for Windows cp1252 consoles
    try:
        logger.info("\n%s", report)
    except UnicodeEncodeError:
        logger.info(report.encode("ascii", errors="replace").decode("ascii"))
    logger.info("Report saved to %s", REPORT_PATH)
    logger.info("Training log saved to %s", REGISTRY_DIR / "training_log.csv")


if __name__ == "__main__":
    main()


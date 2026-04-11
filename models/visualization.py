"""
Model Visualization — Feature Importance + Confusion Matrix.

Generates:
  1. Top-20 Feature Importance bar chart (from RF .feature_importances_)
  2. Confusion matrix heatmap (normalized)
  3. Per-class accuracy bar chart

Usage: python Crop_Recommendation_Engine/models/visualization.py
"""
import json
import sys
import warnings
from pathlib import Path

# Add project root to path for central imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
ENCODERS_JSON = BASE_DIR / "data" / "processed" / "label_encoders.json"
REGISTRY_DIR = BASE_DIR / "models" / "model_registry"
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Central feature engineering (single source of truth) ───
from feature_engineering import META_COLS, TARGET_COL, add_interaction_features


def _write(f, msg=""):
    """Write a line to the report file and flush immediately."""
    f.write(msg + "\n")
    f.flush()


def main():
    outpath = OUTPUT_DIR / "visualization_report.txt"

    with open(outpath, "w", encoding="utf-8") as f:
        try:
            _write(f, "=" * 70)
            _write(f, "MODEL VISUALIZATION REPORT")
            _write(f, "=" * 70)

            # Load data
            df = pd.read_csv(FEATURES_CSV)
            class_counts = df["crop_label"].value_counts()
            rare = class_counts[class_counts < 5].index.tolist()
            if rare:
                df = df[~df["crop_label"].isin(rare)].reset_index(drop=True)
                le = LabelEncoder()
                df[TARGET_COL] = le.fit_transform(df["crop_label"])

            df = add_interaction_features(df)
            feat_cols = [c for c in df.columns
                         if c not in META_COLS and c != TARGET_COL]

            test_mask = df["season_year"] == 2025
            X_test = df.loc[test_mask, feat_cols].values
            y_test = df.loc[test_mask, TARGET_COL].values
            labels = sorted(df["crop_label"].unique())

            # Load model
            model = joblib.load(REGISTRY_DIR / "best_model_2026_03.pkl")
            y_pred = model.predict(X_test)

            # ═══ 1. Feature Importance ═══
            _write(f, "\n[1] Feature Importance (Top 20)")
            importances = model.feature_importances_
            idx_sorted = np.argsort(importances)[::-1][:20]

            fig, ax = plt.subplots(figsize=(10, 8))
            top_names = [feat_cols[i] for i in idx_sorted]
            top_vals = importances[idx_sorted]
            ax.barh(range(len(top_names)), top_vals[::-1], color="forestgreen")
            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names[::-1], fontsize=9)
            ax.set_xlabel("Importance (Gini)")
            ax.set_title("Top 20 Feature Importances - Random Forest")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
            plt.close(fig)
            _write(f, "    Saved: evaluation/feature_importance.png")

            for i, fi in enumerate(idx_sorted[:10]):
                _write(f, f"    {i+1:2d}. {feat_cols[fi]:30s} "
                          f"{importances[fi]:.4f}")

            # ═══ 2. Confusion Matrix ═══
            _write(f, "\n[2] Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            fig, ax = plt.subplots(figsize=(16, 14))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=labels, yticklabels=labels, ax=ax,
                        linewidths=0.5, vmin=0, vmax=1)
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            ax.set_title("Normalized Confusion Matrix - Random Forest",
                         fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
            plt.close(fig)
            _write(f, "    Saved: evaluation/confusion_matrix.png")

            # ═══ 3. Per-class Accuracy ═══
            _write(f, "\n[3] Per-class Accuracy")
            per_class_acc = cm_norm.diagonal()
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ["#e74c3c" if a < 0.85 else "#f39c12" if a < 0.95
                       else "#2ecc71" for a in per_class_acc]
            ax.bar(range(len(labels)), per_class_acc, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Accuracy")
            ax.set_title("Per-class Accuracy - Random Forest")
            ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5,
                        label="85%")
            ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5,
                        label="95%")
            ax.legend()
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=150)
            plt.close(fig)
            _write(f, "    Saved: evaluation/per_class_accuracy.png")

            for lab, acc in zip(labels, per_class_acc):
                flag = ("LOW" if acc < 0.85 else
                        "OK" if acc < 0.95 else "HIGH")
                _write(f, f"    {lab:20s} {acc*100:5.1f}%  [{flag}]")

            # Overall
            overall_acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
            _write(f, f"\n    Overall Accuracy: {overall_acc:.4f}")
            _write(f, f"    F1-macro:        {f1m:.4f}")

            # ═══ 4. Reliability Diagram (Calibration Plot) ═══
            _write(f, "\n[4] Reliability Diagram (Calibration)")
            try:
                cal_dict = joblib.load(
                    REGISTRY_DIR / "calibrator_2026_03.pkl")
                temp = cal_dict["temperature"]

                raw_probs = model.predict_proba(X_test)
                scaled_logits = np.log(raw_probs + 1e-12) / temp
                exp_logits = np.exp(
                    scaled_logits - scaled_logits.max(axis=1, keepdims=True))
                cal_probs = (exp_logits
                             / exp_logits.sum(axis=1, keepdims=True))

                max_cal = cal_probs.max(axis=1)
                max_raw = raw_probs.max(axis=1)
                correct = (y_pred == y_test).astype(float)

                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                for ax, probs, title_label in [
                    (axes[0], max_raw, "Before Calibration (Raw)"),
                    (axes[1], max_cal,
                     "After Calibration (Temperature Scaled)"),
                ]:
                    bin_accs, bin_confs, bin_counts = [], [], []
                    for i in range(n_bins):
                        lo, hi = bin_edges[i], bin_edges[i + 1]
                        mask = ((probs >= lo) & (probs < hi)
                                if i < n_bins - 1
                                else (probs >= lo) & (probs <= hi))
                        if mask.sum() == 0:
                            bin_accs.append(0)
                            bin_confs.append((lo + hi) / 2)
                            bin_counts.append(0)
                        else:
                            bin_accs.append(correct[mask].mean())
                            bin_confs.append(probs[mask].mean())
                            bin_counts.append(mask.sum())

                    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.6,
                           color="steelblue", edgecolor="navy",
                           label="Accuracy in bin")
                    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5,
                            label="Perfect calibration")
                    ax.scatter(bin_confs, bin_accs, color="navy",
                               s=30, zorder=5)
                    ax.set_xlabel("Mean Predicted Confidence")
                    ax.set_ylabel("Fraction of Correct Predictions")
                    ax.set_title(title_label)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.05)
                    ax.legend(loc="lower right")

                    total = sum(bin_counts)
                    ece = (sum(abs(a - c) * n / total
                               for a, c, n in zip(bin_accs, bin_confs,
                                                   bin_counts)
                               if total > 0) if total > 0 else 0)
                    ax.text(0.05, 0.92, f"ECE = {ece:.4f}",
                            transform=ax.transAxes, fontsize=11,
                            bbox=dict(boxstyle="round", facecolor="wheat",
                                      alpha=0.5))

                plt.suptitle("Reliability Diagram - Random Forest",
                             fontsize=14)
                plt.tight_layout()
                fig.savefig(OUTPUT_DIR / "reliability_diagram.png", dpi=150)
                plt.close(fig)
                _write(f, f"    Saved: evaluation/reliability_diagram.png")
                _write(f, f"    Temperature: {temp:.3f}")
            except Exception as cal_err:
                _write(f, f"    Skipped reliability diagram: {cal_err}")

            _write(f, f"\n{'='*70}")
            _write(f, "VISUALIZATION COMPLETE")
            _write(f, f"{'='*70}")

        except Exception as e:
            import traceback
            _write(f, f"\nERROR: {e}")
            traceback.print_exc(file=f)

    print(f"Results written to {outpath}")


if __name__ == "__main__":
    main()


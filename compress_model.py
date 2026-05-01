"""
compress_model.py
=================
Compresses the 2026_05 VotingClassifier model to fit under 90 MB for GitHub.

Strategy (in order of preference):
  1. Re-save with joblib compress=9 (lzma) -- lossless, no accuracy change
  2. If still > 90 MB, prune RF from 600 -> N trees
  3. Save compressed calibrator separately (isotonic maps only, no wrapped model)

Usage:
    python compress_model.py
"""

import os
import copy
import joblib
import numpy as np
from pathlib import Path

REGISTRY  = Path(__file__).parent / "models" / "model_registry" / "2026_05"
MODEL_IN  = REGISTRY / "best_model_2026_05.pkl"
CAL_IN    = REGISTRY / "calibrator_2026_05.pkl"
MODEL_OUT = REGISTRY / "best_model_2026_05_compressed.pkl"
CAL_OUT   = REGISTRY / "calibrator_2026_05_compressed.pkl"
LIMIT_MB  = 90.0

# Small artifacts that the inference loader also looks up via _compressed.pkl
# fallback. Compressing them keeps the GitHub/Streamlit deployment self-contained
# from a single set of *_compressed.pkl files.
SMALL_ARTIFACTS = ["scaler_2026_05.pkl", "ood_stats_2026_05.pkl", "conformal_2026_05.pkl"]

def mb(path):
    return os.path.getsize(path) / 1024 / 1024

def try_compress(model, path, compress_level=9):
    """Save model with lzma compression and return file size in MB."""
    joblib.dump(model, path, compress=("lzma", compress_level))
    size = mb(path)
    print(f"  >> {path.name}: {size:.1f} MB  (compress=lzma,{compress_level})")
    return size

def prune_rf(rf, target_trees):
    """Return a shallow copy of rf with only the first target_trees estimators."""
    rf_small = copy.deepcopy(rf)
    rf_small.estimators_ = rf_small.estimators_[:target_trees]
    rf_small.n_estimators = target_trees
    return rf_small

def main():
    print("=" * 60)
    print("Model Compression Script -- 2026_05")
    print("=" * 60)
    print(f"Input model  : {mb(MODEL_IN):.1f} MB")
    print(f"Input calib  : {mb(CAL_IN):.1f} MB")
    print(f"Target       : < {LIMIT_MB} MB each\n")

    # -- Step 1: Load --
    print("Loading model (may take 1-2 min)...")
    model = joblib.load(MODEL_IN)
    print(f"  VotingClassifier loaded. Estimators: {[n for n, _ in model.estimators]}")

    print("Loading calibrator...")
    cal_data = joblib.load(CAL_IN)
    print(f"  Calibrator method: {cal_data.get('method', '?')}")

    # -- Step 2: Try lossless compression first --
    print("\n[Step 1] Trying lossless lzma compression on model...")
    size_model = try_compress(model, MODEL_OUT, compress_level=9)

    if size_model <= LIMIT_MB:
        print(f"  [OK] Model fits at {size_model:.1f} MB -- lossless compression succeeded!\n")
    else:
        print(f"  [!] Still {size_model:.1f} MB. Pruning RF trees...\n")

        # -- Step 3: Find RF and prune --
        rf_name = None
        rf_orig = None
        for name, est in model.estimators_:
            if hasattr(est, "estimators_"):   # RandomForest
                rf_name = name
                rf_orig  = est
                break

        if rf_orig is None:
            print("  [!] No RandomForest found. Aborting prune.")
        else:
            orig_n = len(rf_orig.estimators_)
            print(f"  RF has {orig_n} trees. Testing pruned sizes...")
            pruned_ok = False

            for target_n in [300, 200, 150, 100]:
                rf_small    = prune_rf(rf_orig, target_n)
                model_small = copy.deepcopy(model)

                # Patch estimators list and named_estimators_ dict
                for i, (n, _) in enumerate(model_small.estimators_):
                    if n == rf_name:
                        model_small.estimators_[i] = (n, rf_small)
                        break
                if rf_name in model_small.named_estimators_:
                    model_small.named_estimators_[rf_name] = rf_small

                test_path  = REGISTRY / f"best_model_2026_05_rf{target_n}.pkl"
                size_small = try_compress(model_small, test_path, compress_level=9)

                if size_small <= LIMIT_MB:
                    print(f"  [OK] RF pruned to {target_n} trees -> {size_small:.1f} MB")
                    if MODEL_OUT.exists():
                        MODEL_OUT.unlink()
                    test_path.rename(MODEL_OUT)
                    model     = model_small
                    pruned_ok = True
                    print(f"  Saved as {MODEL_OUT.name}")
                    break
                else:
                    test_path.unlink()
                    print(f"  [!] {target_n} trees -> {size_small:.1f} MB -- still too large")

            if not pruned_ok:
                print("  [!] Could not get model under target with pruning alone.")

    # -- Step 4: Compress calibrator --
    print("\n[Step 2] Compressing calibrator...")

    method = cal_data.get("method", "temperature")
    if method == "isotonic" and "estimator" in cal_data:
        cal_estimator = cal_data["estimator"]
        print("  Isotonic calibrator detected.")
        print("  Extracting isotonic calibration maps (no wrapped model)...")

        try:
            calibrated_classifiers = cal_estimator.calibrated_classifiers_
            print(f"  Found {len(calibrated_classifiers)} calibrated classifiers (CV folds)")

            iso_maps = []
            for cc in calibrated_classifiers:
                fold_maps = []
                for cal in cc.calibrators:
                    fold_maps.append({
                        "X_thresholds_": cal.X_thresholds_,
                        "y_thresholds_": cal.y_thresholds_,
                        "increasing_":   cal.increasing_,
                    })
                iso_maps.append(fold_maps)

            slim_cal = {
                "method":      "isotonic",
                "temperature": cal_data.get("temperature", 1.0),
                "iso_maps":    iso_maps,
                "classes_":    cal_estimator.classes_,
                "n_classes":   len(cal_estimator.classes_),
                "note":        "slim calibrator -- isotonic maps only, no wrapped model",
            }
            joblib.dump(slim_cal, CAL_OUT, compress=("lzma", 9))
            size_cal = mb(CAL_OUT)
            print(f"  [OK] Slim isotonic calibrator saved: {size_cal:.4f} MB")

        except Exception as e:
            print(f"  [!] Could not extract isotonic maps: {e}")
            print("  Falling back to temperature-only calibration...")
            slim_cal = {
                "method":      "temperature",
                "temperature": cal_data.get("temperature", 1.0),
                "note":        "slim calibrator -- temperature scaling only",
            }
            joblib.dump(slim_cal, CAL_OUT, compress=("lzma", 9))
            size_cal = mb(CAL_OUT)
            print(f"  [OK] Temperature calibrator saved: {size_cal:.4f} MB")
    else:
        slim_cal = {
            "method":      "temperature",
            "temperature": cal_data.get("temperature", 1.0),
        }
        joblib.dump(slim_cal, CAL_OUT, compress=("lzma", 9))
        size_cal = mb(CAL_OUT)
        print(f"  [OK] Temperature calibrator saved: {size_cal:.4f} MB")

    # -- Step 3: Compress small artifacts (scaler, ood_stats, conformal) --
    print("\n[Step 3] Compressing small artifacts (scaler / ood_stats / conformal)...")
    small_results = []
    for fname in SMALL_ARTIFACTS:
        src = REGISTRY / fname
        if not src.exists():
            print(f"  [skip] {fname} not present in registry")
            continue
        dst = REGISTRY / f"{src.stem}_compressed{src.suffix}"
        try:
            data = joblib.load(src)
            joblib.dump(data, dst, compress=("lzma", 9))
            src_kb = src.stat().st_size / 1024
            dst_kb = dst.stat().st_size / 1024
            print(f"  [OK] {fname}: {src_kb:.1f} KB -> {dst.name}: {dst_kb:.1f} KB")
            small_results.append((fname, dst))
        except Exception as e:
            print(f"  [!] Failed to compress {fname}: {e}")

    # -- Summary --
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original model   : {mb(MODEL_IN):.1f} MB")
    if MODEL_OUT.exists():
        print(f"Compressed model : {mb(MODEL_OUT):.1f} MB  ({MODEL_OUT.name})")
    if CAL_OUT.exists():
        print(f"Compressed calib : {mb(CAL_OUT):.4f} MB  ({CAL_OUT.name})")
    for fname, dst in small_results:
        print(f"Compressed small : {dst.stat().st_size / 1024:.2f} KB  ({dst.name})")

    fits = MODEL_OUT.exists() and mb(MODEL_OUT) <= LIMIT_MB
    print(f"\nFits under {LIMIT_MB} MB: {'[YES]' if fits else '[NO]'}")
    print("\nNext steps:")
    print("  1. Commit only the *_compressed.pkl artifacts (originals stay local).")
    print("  2. inference.py auto-prefers *_compressed.pkl over the originals.")
    print("  3. Test with: python test_t9_smoke.py")
    print("  4. Add original *.pkl files to .gitignore")

if __name__ == "__main__":
    main()

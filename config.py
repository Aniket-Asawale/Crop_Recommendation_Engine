"""
Central configuration for the Crop Recommendation Engine.

All constants, paths, thresholds, and mappings live here.
Import from this module instead of hard-coding values in individual scripts.
"""

from pathlib import Path

# ─── Directory Layout ───
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "synthetic" / "crop_recommendation_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
ENCODERS_JSON = PROCESSED_DIR / "label_encoders.json"
REPORT_PATH = PROCESSED_DIR / "model_comparison_report.txt"
PREPROCESSING_REPORT = PROCESSED_DIR / "preprocessing_report.txt"
REGISTRY_DIR = BASE_DIR / "models" / "model_registry"
EVALUATION_DIR = BASE_DIR / "evaluation"

# ─── Model Training ───
MODEL_STAMP = "2026_03"
RANDOM_STATE = 42
LABEL_NOISE_RATE = 0.05
SMOTE_K_NEIGHBORS = 3
TEMPORAL_TRAIN_YEARS = [2023, 2024]
TEMPORAL_TEST_YEAR = 2025
MIN_CLASS_SAMPLES = 5  # drop classes with fewer samples

# ─── Confidence Thresholds ───
CONFIDENCE_HIGH = 0.75
CONFIDENCE_LOW = 0.60
OOD_PROB_THRESHOLD = 0.30  # max(probs) below this = all_low -> OOD

# ─── Maharashtra Bounding Box (training envelope) ───
MH_LAT_RANGE = (15.5, 22.5)
MH_LON_RANGE = (72.5, 80.5)

# ─── Soil Drainage Ordinal Mapping ───
# Sandy drains fastest (1) -> Clay drains slowest (7)
SOIL_DRAINAGE_MAP = {
    "Sandy": 1, "Laterite": 2, "Red": 3, "Alluvial": 4,
    "Medium Black": 5, "Black (Regur)": 6, "Shallow Black": 5, "Clay": 7,
}

# ─── Valid Categorical Values ───
VALID_SEASONS = {"Kharif", "Rabi", "Zaid"}
VALID_SOIL_TYPES = set(SOIL_DRAINAGE_MAP.keys())

# ─── Input Validation Bounds ───
INPUT_BOUNDS = {
    "nitrogen":       (0, 500,   "mg/kg"),
    "phosphorus":     (0, 300,   "mg/kg"),
    "potassium":      (0, 500,   "mg/kg"),
    "temperature":    (0, 55,    "deg C"),
    "moisture":       (0, 100,   "%RH"),
    "ec":             (0, 20000, "uS/cm"),
    "ph":             (3.0, 10.0, ""),
    "weather_temp":   (-5, 55,   "deg C"),
    "humidity":       (0, 100,   "%"),
    "rainfall":       (0, 5000,  "mm"),
    "sunshine":       (0, 14,    "hrs/day"),
    "wind_speed":     (0, 100,   "km/h"),
    "lat":            (5, 40,    "degrees"),
    "lon":            (65, 100,  "degrees"),
    "altitude":       (0, 5000,  "m"),
    "organic_carbon": (0, 10,    "%"),
    "month":          (1, 12,    ""),
}

# ─── Crop Season Mapping (for rotation planner) ───
SEASON_MONTHS = {
    "Kharif": (6, 10),   # June-October
    "Rabi":   (11, 3),   # November-March
    "Zaid":   (3, 6),    # March-June (overlap on boundaries)
}


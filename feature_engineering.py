"""
Centralised feature engineering for the Crop Recommendation Engine.

All interaction / derived features are defined HERE and imported by
baseline_models.py, ann_model.py, visualization.py, and inference.py.
This prevents logic drift between training and inference.
"""

import pandas as pd


# Metadata columns excluded from ML features
META_COLS = [
    "location_id", "city", "state", "season_year",
    "crop_label", "confidence_label", "data_quality_flag",
]
TARGET_COL = "crop_label_encoded"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add agronomic interaction features to improve separability.

    Features added (8 total):
        N_x_P, N_x_K, P_x_K          — NPK cross-products
        temp_x_moisture               — soil temp x soil moisture
        temp_x_humidity               — air temp x air humidity
        rain_x_humidity               — rainfall x humidity
        ph_x_ec                       — pH x electrical conductivity
        moisture_rain_ratio           — soil moisture / (rainfall + 1)
    """
    df = df.copy()
    # NPK interactions
    df["N_x_P"] = df["sensor_nitrogen"] * df["sensor_phosphorus"]
    df["N_x_K"] = df["sensor_nitrogen"] * df["sensor_potassium"]
    df["P_x_K"] = df["sensor_phosphorus"] * df["sensor_potassium"]
    # Climate interactions
    df["temp_x_moisture"] = df["sensor_temperature"] * df["sensor_moisture"]
    df["temp_x_humidity"] = df["weather_temp_mean"] * df["weather_humidity_mean"]
    df["rain_x_humidity"] = df["weather_rainfall_mm"] * df["weather_humidity_mean"]
    # Soil-climate
    df["ph_x_ec"] = df["sensor_ph"] * df["sensor_ec"]
    df["moisture_rain_ratio"] = (
        df["sensor_moisture"] / (df["weather_rainfall_mm"] + 1)
    )
    return df


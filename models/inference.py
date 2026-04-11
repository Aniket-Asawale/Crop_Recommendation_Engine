"""
Top-3 Crop Recommendation Inference Module.

Loads the trained Random Forest model, scaler, and temperature calibrator.
Accepts raw sensor/weather/soil inputs and returns a structured recommendation
with top-3 crops, calibrated confidence, confidence flags, and advisory notes.

Usage:
    from models.inference import CropRecommender
    recommender = CropRecommender()
    result = recommender.predict(
        nitrogen=110, phosphorus=55, potassium=70,
        temperature=26, moisture=55, ec=1400, ph=6.0,
        weather_temp=27, humidity=75, rainfall=900,
        sunshine=4.5, wind_speed=8,
        lat=20.93, lon=77.75, altitude=343,
        organic_carbon=0.67,
        soil_type="Black (Regur)", soil_texture="Clay Loam",
        drainage="Moderate", agro_zone="Vidarbha",
        season="Kharif", month=7,
    )
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path for central imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_engineering import META_COLS, TARGET_COL, add_interaction_features

# ─── Constants (from central config) ───
from config import (
    BASE_DIR, REGISTRY_DIR, FEATURES_CSV, ENCODERS_JSON,
    CONFIDENCE_HIGH as HIGH_THRESHOLD,
    CONFIDENCE_LOW as MEDIUM_THRESHOLD,
    MH_LAT_RANGE, MH_LON_RANGE,
    SOIL_DRAINAGE_MAP, OOD_PROB_THRESHOLD,
    VALID_SEASONS, VALID_SOIL_TYPES, INPUT_BOUNDS,
    MODEL_STAMP,
)




class CropRecommender:
    """Production-ready crop recommendation engine.

    Loads model artifacts once, then provides fast inference
    with calibrated top-3 predictions and confidence flags.
    """

    def __init__(self, model_stamp: Optional[str] = None):
        """Load model, scaler, calibrator, encoders, and feature column order.

        If model_stamp is None, auto-discovers the latest model in model_registry
        by sorting best_model_*.pkl files and picking the most recent.
        """
        if model_stamp is None:
            model_stamp = self._discover_latest_stamp()
            logger.info("Auto-discovered model stamp: %s", model_stamp)

        self.model_stamp = model_stamp
        self.model = joblib.load(REGISTRY_DIR / f"best_model_{model_stamp}.pkl")
        self.scaler = joblib.load(REGISTRY_DIR / f"scaler_{model_stamp}.pkl")
        cal_data = joblib.load(REGISTRY_DIR / f"calibrator_{model_stamp}.pkl")
        self.temperature = cal_data["temperature"]
        logger.info("Loaded model=%s, T=%.3f", model_stamp, self.temperature)

        with open(ENCODERS_JSON) as f:
            self.encoders = json.load(f)

        # Reverse map: encoded int -> crop name
        self.crop_labels = {v: k for k, v in self.encoders["crop_label"].items()}

        # Get feature column order from training data
        df_ref = pd.read_csv(FEATURES_CSV, nrows=5)
        df_ref = add_interaction_features(df_ref)
        self.feat_cols = [c for c in df_ref.columns
                         if c not in META_COLS and c != TARGET_COL]
        logger.debug("Feature columns (%d): %s", len(self.feat_cols), self.feat_cols[:5])

    @staticmethod
    def _discover_latest_stamp() -> str:
        """Auto-discover the latest model stamp from model_registry."""
        model_files = sorted(REGISTRY_DIR.glob("best_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {REGISTRY_DIR}. "
                "Run baseline_models.py first to train a model."
            )
        # Extract stamp from filename: best_model_YYYY_MM.pkl -> YYYY_MM
        latest = model_files[-1].stem.replace("best_model_", "")
        return latest

    def _calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling calibration."""
        logits = np.log(np.clip(raw_probs, 1e-10, 1.0))
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def _encode_categorical(self, category: str, encoder_key: str) -> int:
        """Safely encode a categorical value, defaulting to 0 if unknown."""
        mapping = self.encoders.get(encoder_key, {})
        return mapping.get(category, 0)

    def _is_out_of_distribution(self, lat: float, lon: float,
                                 cal_probs: np.ndarray) -> bool:
        """Check if input is outside Maharashtra training envelope.

        OOD is triggered only when the location is geographically outside
        Maharashtra.  Low-confidence predictions inside Maharashtra are
        legitimate (e.g. rare soil-season combos) and should NOT be
        flagged as OOD — they already surface via the UNCERTAIN flag.
        """
        geo_outside = (lat < MH_LAT_RANGE[0] or lat > MH_LAT_RANGE[1] or
                       lon < MH_LON_RANGE[0] or lon > MH_LON_RANGE[1])
        return geo_outside

    def _confidence_flag(self, confidence: float) -> str:
        """Assign confidence tier."""
        if confidence >= HIGH_THRESHOLD:
            return "HIGH"
        elif confidence >= MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    # Input validation bounds and valid values imported from config module

    def _validate_inputs(self, **kwargs) -> list:
        """Validate input values and return list of warnings/errors.

        Raises ValueError for missing required fields or out-of-range values.
        Returns list of warning strings for soft-boundary issues.
        """
        required = list(INPUT_BOUNDS.keys()) + [
            "soil_type", "soil_texture", "drainage", "agro_zone", "season",
        ]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        warnings_list = []

        # Numeric bounds
        for field, (lo, hi, unit) in INPUT_BOUNDS.items():
            val = kwargs[field]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"'{field}' must be numeric, got {type(val).__name__}: {val}"
                )
            if val < lo or val > hi:
                raise ValueError(
                    f"'{field}' = {val} is out of range [{lo}, {hi}] {unit}"
                )

        # Categorical checks
        season = kwargs.get("season", "")
        if season not in VALID_SEASONS:
            raise ValueError(
                f"'season' must be one of {VALID_SEASONS}, got '{season}'"
            )

        soil = kwargs.get("soil_type", "")
        if soil not in VALID_SOIL_TYPES:
            warnings_list.append(
                f"Unknown soil_type '{soil}' — defaulting encoder to 0. "
                f"Known types: {sorted(VALID_SOIL_TYPES)}"
            )

        # Soft warnings for Maharashtra-specific plausibility
        lat, lon = kwargs.get("lat", 0), kwargs.get("lon", 0)
        if lat < MH_LAT_RANGE[0] or lat > MH_LAT_RANGE[1] or \
           lon < MH_LON_RANGE[0] or lon > MH_LON_RANGE[1]:
            warnings_list.append(
                f"Location ({lat:.2f}, {lon:.2f}) is outside Maharashtra "
                f"training envelope. Prediction may be unreliable."
            )

        # Soil-drainage consistency check
        from generators.crop_profiles import SOIL_DRAINAGE_VALID
        soil = kwargs.get("soil_type", "")
        drainage = kwargs.get("drainage", "")
        if soil in SOIL_DRAINAGE_VALID and drainage:
            valid_drainages = SOIL_DRAINAGE_VALID[soil]
            if drainage not in valid_drainages:
                warnings_list.append(
                    f"⚠️ Inconsistent inputs: '{soil}' soil typically has "
                    f"{valid_drainages} drainage, but '{drainage}' was provided. "
                    f"This combination is physically unlikely and may reduce "
                    f"prediction accuracy."
                )

        # Season-month consistency check
        from generators.crop_profiles import SEASON_MONTHS as SEASON_MONTH_MAP
        month = kwargs.get("month", None)
        if month is not None and season:
            valid_months = SEASON_MONTH_MAP.get(season, [])
            if valid_months and int(month) not in valid_months:
                warnings_list.append(
                    f"⚠️ Season-month mismatch: '{season}' season typically "
                    f"covers months {valid_months}, but month={int(month)} was "
                    f"provided. This may indicate incorrect input."
                )

        # Agro-zone vs location consistency check
        from generators.crop_profiles import AGRO_ZONE_BOUNDS
        agro_zone = kwargs.get("agro_zone", "")
        if agro_zone in AGRO_ZONE_BOUNDS:
            bounds = AGRO_ZONE_BOUNDS[agro_zone]
            lat_lo, lat_hi = bounds["lat"]
            lon_lo, lon_hi = bounds["lon"]
            if lat < lat_lo or lat > lat_hi or lon < lon_lo or lon > lon_hi:
                warnings_list.append(
                    f"⚠️ Zone-location mismatch: coordinates ({lat:.2f}, "
                    f"{lon:.2f}) are outside the typical bounds of "
                    f"'{agro_zone}' (lat {lat_lo}-{lat_hi}, lon "
                    f"{lon_lo}-{lon_hi}). Check agro_zone or coordinates."
                )

        return warnings_list

    # Irrigation multipliers for effective moisture/rainfall
    _IRRIGATION_MULTIPLIERS = {
        "Rainfed":   {"moisture": 1.0, "rainfall": 1.0},
        "Drip":      {"moisture": 1.3, "rainfall": 1.4},   # efficient supplemental
        "Sprinkler": {"moisture": 1.2, "rainfall": 1.3},
        "Flood":     {"moisture": 1.5, "rainfall": 1.6},   # canal/flood irrigation
    }

    def _build_feature_vector(self, **kwargs) -> np.ndarray:
        """Convert raw inputs into a feature vector matching training order."""
        N = kwargs["nitrogen"]
        P = kwargs["phosphorus"]
        K = kwargs["potassium"]
        temp = kwargs["temperature"]
        moist = kwargs["moisture"]
        ec = kwargs["ec"]
        ph = kwargs["ph"]
        wt = kwargs["weather_temp"]
        hum = kwargs["humidity"]
        rain = kwargs["rainfall"]
        sun = kwargs["sunshine"]
        wind = kwargs["wind_speed"]
        lat = kwargs["lat"]
        lon = kwargs["lon"]
        alt = kwargs["altitude"]
        oc = kwargs["organic_carbon"]
        season = kwargs["season"]
        month = kwargs["month"]

        # Adjust moisture/rainfall for irrigation type
        irrigation = kwargs.get("irrigation_type", "Rainfed")
        irr_mult = self._IRRIGATION_MULTIPLIERS.get(irrigation, {"moisture": 1.0, "rainfall": 1.0})
        moist = moist * irr_mult["moisture"]
        rain = rain * irr_mult["rainfall"]

        # Encode categoricals
        dr = self._encode_categorical(kwargs["drainage"], "drainage_class")
        az = self._encode_categorical(kwargs["agro_zone"], "agro_zone")
        # NOTE: soil_type and soil_texture are now one-hot encoded (see base dict below)

        # Soil drainage ordinal (from config.py)
        sdo = SOIL_DRAINAGE_MAP.get(kwargs["soil_type"], 4)

        # Season one-hot
        kh = 1 if season == "Kharif" else 0
        rb = 1 if season == "Rabi" else 0
        zd = 1 if season == "Zaid" else 0

        # Month cyclical
        ms = math.sin(2 * math.pi * month / 12)
        mc = math.cos(2 * math.pi * month / 12)

        # Build feature dict (npk_ratio_* and altitude_temp_correction removed — redundant)
        base = {
            "sensor_nitrogen": N, "sensor_phosphorus": P, "sensor_potassium": K,
            "sensor_temperature": temp, "sensor_moisture": moist,
            "sensor_ec": ec, "sensor_ph": ph,
            "weather_temp_mean": wt, "weather_humidity_mean": hum,
            "weather_rainfall_mm": rain, "weather_sunshine_hrs": sun,
            "weather_wind_speed": wind,
            "lat": lat, "lon": lon, "altitude_m": alt,
            "organic_carbon_pct": oc,
            # Irrigation signal (0/1 — key for Rice/Sugarcane vs dryland)
            "irrigation_available": kwargs.get("irrigation_available", 0),
            # Engineered
            "moisture_deficit": moist - rain * 0.1,
            "ec_stress_flag": 1 if ec > 4000 else 0,
            "month_sin": ms, "month_cos": mc,
            "soil_drainage_ordinal": sdo,
            # Encoded categoricals (drainage + agro_zone label-encoded)
            "drainage_class_encoded": dr, "agro_zone_encoded": az,
            "is_season_Kharif": kh, "is_season_Rabi": rb, "is_season_Zaid": zd,
            # Interaction features
            "N_x_P": N * P, "N_x_K": N * K, "P_x_K": P * K,
            "temp_x_moisture": temp * moist,
            "temp_x_humidity": wt * hum,
            "rain_x_humidity": rain * hum,
            "ph_x_ec": ph * ec,
            "moisture_rain_ratio": moist / (rain + 1),
        }

        # One-hot encode soil_type and soil_texture
        soil_type_val = kwargs["soil_type"]
        soil_texture_val = kwargs["soil_texture"]
        base[f"soil_type_{soil_type_val}"] = 1
        base[f"soil_texture_{soil_texture_val}"] = 1
        # All other soil_type_* and soil_texture_* columns default to 0.0
        # via the base.get(c, 0.0) fallback below.

        return np.array([base.get(c, 0.0) for c in self.feat_cols],
                        dtype=np.float64)

    def predict(self, **kwargs) -> dict:
        """Generate top-3 crop recommendation with confidence flags.

        Parameters
        ----------
        nitrogen, phosphorus, potassium : float
            Soil NPK in mg/kg (from sensor)
        temperature : float
            Soil temperature in °C
        moisture : float
            Soil moisture in %RH
        ec : float
            Electrical conductivity in μS/cm
        ph : float
            Soil pH
        weather_temp : float
            Air temperature in °C
        humidity : float
            Relative humidity in %
        rainfall : float
            Seasonal rainfall in mm
        sunshine : float
            Sunshine hours per day
        wind_speed : float
            Wind speed in km/h
        lat, lon : float
            GPS coordinates
        altitude : float
            Elevation in meters
        organic_carbon : float
            Soil organic carbon %
        soil_type : str
            e.g. "Black (Regur)", "Red", "Alluvial", "Laterite", "Sandy"
        soil_texture : str
            e.g. "Clay Loam", "Sandy Loam", "Silty Clay"
        drainage : str
            e.g. "Moderate", "Good", "Poor", "Excessive"
        agro_zone : str
            e.g. "Vidarbha", "Marathwada", "Western Maharashtra"
        season : str
            "Kharif", "Rabi", or "Zaid"
        month : int
            Month number (1-12)
        prev_crop : str, optional
            Name of the crop grown in the previous season. When provided,
            rotation bonuses/penalties are applied to re-rank the top-3.
        irrigation_type : str, optional
            "Rainfed" (default), "Drip", "Sprinkler", or "Flood".
            Adjusts effective moisture/rainfall for crop suitability.

        Returns
        -------
        dict with keys:
            top_3: list of {crop, confidence, season, flag}
            confidence_flag: str (HIGH/MEDIUM/LOW/OUT_OF_DISTRIBUTION)
            advisory: str
            is_ood: bool
        """
        prev_crop = kwargs.pop("prev_crop", None)

        # Validate inputs (raises ValueError on hard errors)
        input_warnings = self._validate_inputs(**kwargs)
        for w in input_warnings:
            logger.warning("Input warning: %s", w)

        # crop_family_encoded removed from training — no ensembling needed
        X = self._build_feature_vector(**kwargs).reshape(1, -1)
        raw_probs = self.model.predict_proba(X)
        cal_probs = self._calibrate(raw_probs)

        # ── Season-aware candidate selection ──
        # HARD RULE: Only crops that belong to the input season are eligible.
        # This prevents Rabi crops from appearing in Kharif predictions, etc.
        from generators.crop_profiles import CROP_TO_SEASON
        input_season = kwargs["season"]

        # Build season-filtered candidate list (take more than 3 to allow
        # backfilling after guardrails drop zero-confidence entries)
        all_indices = cal_probs[0].argsort()[::-1]  # sorted desc by prob
        candidates = []
        for idx in all_indices:
            crop_name = self.crop_labels.get(idx, f"Unknown_{idx}")
            crop_season = CROP_TO_SEASON.get(crop_name, "")
            if crop_season != input_season:
                continue  # hard season filter — skip wrong-season crops
            conf = float(cal_probs[0][idx])
            candidates.append({
                "crop": crop_name,
                "confidence": round(conf, 4),
                "confidence_pct": f"{conf * 100:.1f}%",
                "flag": self._confidence_flag(conf),
                "season": input_season,
            })
            if len(candidates) >= 8:  # enough candidates for guardrail filtering
                break

        # Apply agronomic guardrails (soil/EC/drainage/regional adjustments)
        candidates = self._apply_agronomic_guardrails(candidates, kwargs)

        # Apply rotation bonus/penalty if prev_crop is provided
        if prev_crop is not None:
            candidates = self._apply_rotation_adjustment(candidates, prev_crop)

        # Filter out zero-confidence crops and take top 3
        top3 = [c for c in candidates if c["confidence"] > 0.001]
        if len(top3) < 1:
            top3 = candidates[:3]  # fallback: show best even if zero
        else:
            top3 = top3[:3]

        # Overall flag = flag of top-1
        top1_conf = top3[0]["confidence"]
        overall_flag = self._confidence_flag(top1_conf)

        # OOD check
        is_ood = self._is_out_of_distribution(
            kwargs["lat"], kwargs["lon"], cal_probs[0]
        )
        if is_ood:
            overall_flag = "OUT_OF_DISTRIBUTION"
            logger.warning("OOD detected for (%s, %s)", kwargs["lat"], kwargs["lon"])

        # Advisory text
        advisory = self._build_advisory(top3, overall_flag, is_ood, kwargs)

        # Structured farmer advisory (for UI panel + mobile API)
        farmer_advisory = self._build_farmer_advisory(top3, overall_flag, is_ood, kwargs)

        return {
            "top_3": top3,
            "confidence_flag": overall_flag,
            "advisory": advisory,
            "farmer_advisory": farmer_advisory,
            "is_ood": is_ood,
            "input_warnings": input_warnings,
        }

    # ──────────────────────────────────────────────────────────────
    # Agronomic guardrails (post-prediction correction)
    # ──────────────────────────────────────────────────────────────

    def _apply_agronomic_guardrails(self, top3: list, inputs: dict) -> list:
        """Apply soil, EC, drainage, and regional guardrails.

        Adjusts confidence scores of top-3 predictions based on agronomic
        hard-rules that the ML model may not have learned from synthetic data.

        Penalties/bonuses:
          - Soil incompatibility: −15% confidence
          - EC sensitivity breach: −10% confidence (progressive)
          - Drainage incompatibility: −20% for sensitive crops on poor drainage
          - Drainage boost: +10% for tolerant crops on poor drainage
          - Regional dominance boost: +5% for zone staple crops
        """
        from generators.crop_profiles import (
            SOIL_CROP_INCOMPATIBLE, EC_SENSITIVE_CROPS,
            REGIONAL_CROP_DOMINANCE,
            DRAINAGE_TOLERANT_CROPS, DRAINAGE_SENSITIVE_CROPS,
        )

        soil_type = inputs.get("soil_type", "")
        ec = inputs.get("ec", 0)
        agro_zone = inputs.get("agro_zone", "")
        season = inputs.get("season", "")
        drainage = inputs.get("drainage", "")

        incompatible = set(SOIL_CROP_INCOMPATIBLE.get(soil_type, []))
        zone_priors = REGIONAL_CROP_DOMINANCE.get(agro_zone, {}).get(season, {})
        is_poor_drainage = drainage in ("Poor", "Very Poor")

        adjusted = []
        for crop_info in top3:
            crop_name = crop_info["crop"]
            adj = 0.0
            notes = []

            # 1. Soil incompatibility penalty
            if crop_name in incompatible:
                adj -= 0.15
                notes.append(f"soil penalty: {crop_name} unsuited for {soil_type}")

            # 2. EC sensitivity penalty
            if crop_name in EC_SENSITIVE_CROPS:
                threshold = EC_SENSITIVE_CROPS[crop_name]
                if ec > threshold:
                    ec_pen = min(0.10, 0.10 * (ec - threshold) / threshold)
                    adj -= ec_pen
                    notes.append(f"EC penalty: {crop_name} sensitive above {threshold} μS/cm")

            # 3. Drainage-based adjustments (critical for Konkan/waterlogged areas)
            if is_poor_drainage:
                if crop_name in DRAINAGE_TOLERANT_CROPS:
                    adj += 0.15
                    notes.append(
                        f"drainage boost: {crop_name} thrives in {drainage} drainage"
                    )
                elif crop_name in DRAINAGE_SENSITIVE_CROPS:
                    # Scale penalty to reduce confidence significantly
                    adj -= 0.30
                    notes.append(
                        f"drainage penalty: {crop_name} fails in {drainage} drainage "
                        f"(root rot / waterlogging risk)"
                    )

            # 4. Regional dominance boost (scaled by dominance weight)
            if crop_name in zone_priors:
                weight = zone_priors[crop_name]
                if weight >= 2.0:
                    adj += 0.15  # primary regional staple
                    notes.append(f"regional boost: {crop_name} is a primary {agro_zone} staple")
                elif weight >= 1.5:
                    adj += 0.08  # secondary regional crop
                    notes.append(f"regional boost: {crop_name} is a {agro_zone} staple")

            adj_conf = min(1.0, max(0.0, crop_info["confidence"] + adj))
            adjusted.append({
                **crop_info,
                "confidence": round(adj_conf, 4),
                "confidence_pct": f"{adj_conf * 100:.1f}%",
                "flag": self._confidence_flag(adj_conf),
                "guardrail_notes": notes,
            })

        adjusted.sort(key=lambda x: x["confidence"], reverse=True)
        return adjusted

    def _build_advisory(self, top3: list, flag: str,
                        is_ood: bool, inputs: dict) -> str:
        """Generate human-readable advisory note."""
        crop1 = top3[0]["crop"]
        conf1 = top3[0]["confidence_pct"]

        if is_ood:
            return (
                f"WARNING: Location ({inputs['lat']:.2f}, {inputs['lon']:.2f}) "
                f"is outside the model's training region (Maharashtra). "
                f"Prediction ({crop1} at {conf1}) may be unreliable. "
                f"Consult local agricultural experts."
            )

        if flag == "HIGH":
            return (
                f"Strong recommendation: {crop1} ({conf1} confidence). "
                f"Well-suited for {inputs['soil_type']} soil in "
                f"{inputs['agro_zone']} during {inputs['season']} season."
            )

        if flag == "MEDIUM":
            crop2 = top3[1]["crop"] if len(top3) > 1 else "N/A"
            return (
                f"Moderate confidence: {crop1} ({conf1}). "
                f"Also consider {crop2} ({top3[1]['confidence_pct']}). "
                f"Local soil testing recommended for final decision."
            )

        # LOW
        alternatives = ", ".join(f"{t['crop']} ({t['confidence_pct']})"
                                 for t in top3)
        return (
            f"Low confidence prediction. Top options: {alternatives}. "
            f"Consult your local Krishi Vigyan Kendra (KVK) for "
            f"{inputs['agro_zone']} region-specific advice."
        )

    # ──────────────────────────────────────────────────────────────
    # Structured farmer advisory (for UI panel + mobile API)
    # ──────────────────────────────────────────────────────────────

    def _build_farmer_advisory(self, top3: list, flag: str,
                               is_ood: bool, inputs: dict) -> dict:
        """Generate a structured farmer-friendly advisory dictionary.

        Returns a dict with keys:
            why_this_crop: str — plain-language reason for the recommendation
            warnings: list[str] — actionable warnings from guardrails
            irrigation_tips: str — soil-based irrigation guidance
            next_crop: str — what to plant next season
            sowing_window: str — when to sow/harvest
            soil_health: dict — quick soil parameter assessment
        """
        from generators.crop_profiles import (
            ALL_CROPS, CROP_TO_FAMILY, CROP_TO_SEASON,
            SOIL_PROFILES, REGIONAL_CROP_DOMINANCE,
            EC_SENSITIVE_CROPS, SEASON_MONTHS,
        )

        crop1 = top3[0]["crop"]
        conf1 = top3[0]["confidence_pct"]
        soil_type = inputs.get("soil_type", "")
        agro_zone = inputs.get("agro_zone", "")
        season = inputs.get("season", "")
        ec = inputs.get("ec", 0)
        ph = inputs.get("ph", 7.0)

        # ── Why this crop? ──
        reasons = []
        crop_season = CROP_TO_SEASON.get(crop1, season)
        crop_profile = ALL_CROPS.get(crop_season, {}).get(crop1, {})

        soil_affinity = crop_profile.get("soil_affinity", [])
        if soil_type in soil_affinity:
            reasons.append(f"{crop1} thrives in {soil_type} soil (primary affinity)")
        elif soil_type in crop_profile.get("soil_secondary", []):
            reasons.append(f"{crop1} grows well in {soil_type} soil (secondary fit)")

        zone_crops = REGIONAL_CROP_DOMINANCE.get(agro_zone, {}).get(season, {})
        if crop1 in zone_crops:
            weight = zone_crops[crop1]
            if weight >= 2.0:
                reasons.append(f"It is a primary staple crop of {agro_zone}")
            elif weight >= 1.5:
                reasons.append(f"It is commonly grown in {agro_zone}")

        family = CROP_TO_FAMILY.get(crop1, "Unknown")
        reasons.append(f"Crop family: {family} — suited for {season} season")

        if not reasons:
            reasons.append(f"Model prediction based on soil, weather, and location data")

        why = f"{crop1} ({conf1} confidence). " + ". ".join(reasons) + "."

        # ── Warnings (from guardrail notes) ──
        warnings = []
        for crop_info in top3:
            g_notes = crop_info.get("guardrail_notes", [])
            for note in g_notes:
                crop_name = crop_info["crop"]
                if "penalty" in note:
                    # Convert internal note to farmer-friendly warning
                    if "soil" in note:
                        warnings.append(
                            f"⚠️ {crop_name} may struggle in {soil_type} soil — "
                            f"consider a different crop or soil amendment"
                        )
                    elif "EC" in note:
                        threshold = EC_SENSITIVE_CROPS.get(crop_name, 0)
                        warnings.append(
                            f"⚠️ {crop_name} is sensitive to salinity — your EC "
                            f"({ec} μS/cm) exceeds its tolerance ({threshold} μS/cm). "
                            f"Consider gypsum application or selecting a salt-tolerant crop"
                        )
                    elif "drainage" in note:
                        warnings.append(
                            f"⚠️ {crop_name} needs well-drained soil — "
                            f"improve drainage or choose a waterlogging-tolerant crop"
                        )
                    elif "rotation" in note:
                        warnings.append(f"🔄 {note}")

        if is_ood:
            warnings.insert(0,
                f"🚨 Location is outside Maharashtra training region — "
                f"predictions may be unreliable. Consult local KVK.")

        # ── Irrigation guidance ──
        soil_info = SOIL_PROFILES.get(soil_type, {})
        water_ret = soil_info.get("water_retention", 0.5)
        drainage_rate = soil_info.get("drainage_rate", 0.5)

        if water_ret >= 0.80:
            irr_tip = (f"{soil_type} soil has high water retention ({water_ret:.0%}). "
                       f"Irrigate less frequently — risk of waterlogging. "
                       f"Allow soil to dry between irrigations.")
        elif water_ret >= 0.50:
            irr_tip = (f"{soil_type} soil has moderate water retention ({water_ret:.0%}). "
                       f"Standard irrigation schedule. Monitor moisture during dry spells.")
        else:
            irr_tip = (f"{soil_type} soil has low water retention ({water_ret:.0%}). "
                       f"Irrigate frequently in smaller amounts. "
                       f"Consider mulching to reduce evaporation.")

        # Crop-specific water needs
        rainfall_range = crop_profile.get("rainfall_mm", (0, 0))
        if rainfall_range[1] > 0:
            irr_tip += (f" {crop1} needs {rainfall_range[0]}–{rainfall_range[1]} mm "
                        f"total water during the growing season.")

        # ── Next crop suggestion ──
        preferred_next = self._ROTATION_BONUS.get(family, [])
        if preferred_next:
            next_season_map = {"Kharif": "Rabi", "Rabi": "Zaid", "Zaid": "Kharif"}
            next_season = next_season_map.get(season, "Rabi")
            next_season_crops = ALL_CROPS.get(next_season, {})
            suggestions = []
            for next_crop, profile in next_season_crops.items():
                next_family = CROP_TO_FAMILY.get(next_crop, "")
                if next_family in preferred_next and soil_type in (
                    profile.get("soil_affinity", []) + profile.get("soil_secondary", [])
                ):
                    suggestions.append(next_crop)
            if suggestions:
                next_crop_text = (
                    f"After {crop1} ({family}), plant a {'/'.join(preferred_next)} "
                    f"crop in {next_season}. Good options: {', '.join(suggestions[:3])}"
                )
            else:
                next_crop_text = (
                    f"After {crop1} ({family}), prefer {'/'.join(preferred_next)} "
                    f"family crops in the next season for soil health"
                )
        else:
            next_crop_text = "Rotate with a different crop family next season."

        # ── Sowing window ──
        season_months = SEASON_MONTHS.get(season, [])
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        if season_months:
            sow_start = month_names.get(season_months[0], "?")
            sow_end = month_names.get(season_months[-1], "?")
            current_month = inputs.get("month", 0)
            if current_month in season_months:
                timing = f"✅ Current month is within the {season} window"
            else:
                timing = f"📅 {season} sowing window: {sow_start}–{sow_end}"
            sowing = f"{season} season: {sow_start} to {sow_end}. {timing}"
        else:
            sowing = f"Refer to local agricultural calendar for {season} timing."

        # ── Soil health quick assessment ──
        n = inputs.get("nitrogen", 0)
        p = inputs.get("phosphorus", 0)
        k = inputs.get("potassium", 0)

        def _grade(val, low, high):
            if val < low:
                return "Low"
            elif val > high:
                return "High"
            return "Adequate"

        n_range = crop_profile.get("n_range", (80, 150))
        p_range = crop_profile.get("p_range", (40, 80))
        k_range = crop_profile.get("k_range", (60, 130))

        soil_health = {
            "nitrogen": {"value": n, "status": _grade(n, n_range[0], n_range[1]),
                         "ideal_range": f"{n_range[0]}–{n_range[1]} mg/kg"},
            "phosphorus": {"value": p, "status": _grade(p, p_range[0], p_range[1]),
                           "ideal_range": f"{p_range[0]}–{p_range[1]} mg/kg"},
            "potassium": {"value": k, "status": _grade(k, k_range[0], k_range[1]),
                          "ideal_range": f"{k_range[0]}–{k_range[1]} mg/kg"},
            "ph": {"value": ph, "status": _grade(ph, crop_profile.get("ph_range", (6, 8))[0],
                                                  crop_profile.get("ph_range", (6, 8))[1]),
                   "ideal_range": f"{crop_profile.get('ph_range', (6, 8))[0]}–{crop_profile.get('ph_range', (6, 8))[1]}"},
            "ec": {"value": ec, "status": "Concern" if crop1 in EC_SENSITIVE_CROPS and ec > EC_SENSITIVE_CROPS[crop1] else "OK"},
        }

        return {
            "why_this_crop": why,
            "warnings": warnings,
            "irrigation_tips": irr_tip,
            "next_crop": next_crop_text,
            "sowing_window": sowing,
            "soil_health": soil_health,
        }

    # ──────────────────────────────────────────────────────────────
    # Rotation history adjustment
    # ──────────────────────────────────────────────────────────────

    def _apply_rotation_adjustment(self, top3: list, prev_crop: str) -> list:
        """Re-rank top-3 predictions based on rotation with prev_crop.

        Applies +5% bonus for agronomically beneficial succession and
        -5% penalty for same-family repetition.
        """
        from generators.crop_profiles import CROP_TO_FAMILY

        prev_family = CROP_TO_FAMILY.get(prev_crop, "Unknown")
        preferred = self._ROTATION_BONUS.get(prev_family, [])

        adjusted = []
        for crop_info in top3:
            crop_name = crop_info["crop"]
            family = CROP_TO_FAMILY.get(crop_name, "Unknown")
            bonus = 0.0
            note = ""

            if family in preferred:
                bonus = 0.05
                note = f"rotation bonus: {family} after {prev_family}"
            elif family == prev_family:
                bonus = -0.05
                note = f"rotation penalty: same family ({family}) repeated"

            adj_conf = min(1.0, max(0.0, crop_info["confidence"] + bonus))
            adjusted.append({
                **crop_info,
                "confidence": round(adj_conf, 4),
                "confidence_pct": f"{adj_conf * 100:.1f}%",
                "flag": self._confidence_flag(adj_conf),
                "rotation_note": note,
            })

        adjusted.sort(key=lambda x: x["confidence"], reverse=True)
        return adjusted

    # ──────────────────────────────────────────────────────────────
    # Multi-season rotation planner
    # ──────────────────────────────────────────────────────────────

    # Agronomic rotation preferences: family_just_planted → best_next_families
    _ROTATION_BONUS = {
        "Cereal":    ["Legume", "Oilseed"],      # legume fixes N after cereal
        "Legume":    ["Cereal", "Cash"],           # cereal benefits from N
        "Oilseed":   ["Legume", "Cereal"],
        "Cash":      ["Legume", "Cereal"],
        "Vegetable": ["Legume", "Cereal", "Oilseed"],
    }

    # Typical seasonal weather adjustments for Maharashtra
    _SEASON_WEATHER = {
        "Kharif": {"weather_temp": 28, "humidity": 78, "rainfall": 900,
                   "sunshine": 4.5, "wind_speed": 8, "month": 7,
                   "temperature": 27, "moisture": 60},
        "Rabi":   {"weather_temp": 22, "humidity": 55, "rainfall": 350,
                   "sunshine": 7.0, "wind_speed": 6, "month": 12,
                   "temperature": 20, "moisture": 35},
        "Zaid":   {"weather_temp": 34, "humidity": 40, "rainfall": 120,
                   "sunshine": 9.0, "wind_speed": 10, "month": 4,
                   "temperature": 33, "moisture": 20},
    }

    def plan_rotation(self, **kwargs) -> dict:
        """Generate a full-year crop rotation plan: Kharif → Rabi → Zaid.

        Accepts the same base inputs as ``predict()`` (soil, location, etc.).
        Season-specific weather values are auto-filled from typical Maharashtra
        averages but can be overridden per season via ``season_overrides``.

        Parameters
        ----------
        **kwargs :
            Same inputs as ``predict()`` — soil_type, lat, lon, altitude,
            nitrogen, phosphorus, potassium, ec, ph, organic_carbon,
            soil_texture, drainage, agro_zone.
            Season/weather fields are auto-filled per season.
        season_overrides : dict[str, dict], optional
            Per-season overrides, e.g.
            ``{"Kharif": {"rainfall": 1200}, "Rabi": {"humidity": 45}}``

        Returns
        -------
        dict with keys:
            rotation: list of 3 dicts (one per season) each containing
                      season, recommendation (predict() output), and
                      rotation_note (agronomic rationale).
            summary: str — one-line summary of the plan.
        """
        from generators.crop_profiles import CROP_TO_FAMILY, ALL_CROPS

        season_overrides = kwargs.pop("season_overrides", {})
        season_order = ["Kharif", "Rabi", "Zaid"]
        rotation = []
        prev_family = None

        for season in season_order:
            # Build season-specific inputs
            season_inputs = dict(kwargs)
            season_inputs["season"] = season
            # Apply typical weather defaults
            season_inputs.update(self._SEASON_WEATHER[season])
            # Apply user overrides for this season
            if season in season_overrides:
                season_inputs.update(season_overrides[season])

            # Get prediction
            result = self.predict(**season_inputs)
            top3 = result["top_3"]

            # Apply rotation bonus/penalty to re-rank
            if prev_family is not None:
                preferred = self._ROTATION_BONUS.get(prev_family, [])
                scored = []
                for crop_info in top3:
                    crop_name = crop_info["crop"]
                    family = CROP_TO_FAMILY.get(crop_name, "Unknown")
                    bonus = 0.0
                    note = ""

                    # Bonus for agronomically preferred succession
                    if family in preferred:
                        bonus = 0.05
                        note = f"rotation bonus (+5%): {family} after {prev_family}"
                    # Penalty for same family back-to-back
                    elif family == prev_family:
                        bonus = -0.05
                        note = f"rotation penalty (-5%): same family ({family}) repeated"

                    adjusted_conf = min(1.0, max(0.0, crop_info["confidence"] + bonus))
                    scored.append({
                        **crop_info,
                        "confidence": round(adjusted_conf, 4),
                        "confidence_pct": f"{adjusted_conf * 100:.1f}%",
                        "flag": self._confidence_flag(adjusted_conf),
                        "rotation_note": note,
                    })
                # Re-sort by adjusted confidence
                scored.sort(key=lambda x: x["confidence"], reverse=True)
                top3 = scored
                result["top_3"] = top3

            # Determine family of recommended crop for next iteration
            chosen_crop = top3[0]["crop"]
            prev_family = CROP_TO_FAMILY.get(chosen_crop, "Unknown")

            # Check the recommended crop actually belongs to this season
            season_crops = set(ALL_CROPS.get(season, {}).keys())
            in_season = chosen_crop in season_crops
            season_note = ("" if in_season
                           else f"Note: {chosen_crop} is atypical for {season} season.")

            rotation.append({
                "season": season,
                "recommendation": result,
                "chosen_crop": chosen_crop,
                "crop_family": prev_family,
                "in_season": in_season,
                "season_note": season_note,
            })

        # Build summary
        plan_crops = [r["chosen_crop"] for r in rotation]
        plan_families = [r["crop_family"] for r in rotation]
        summary = (
            f"Rotation plan: {plan_crops[0]} (Kharif, {plan_families[0]}) → "
            f"{plan_crops[1]} (Rabi, {plan_families[1]}) → "
            f"{plan_crops[2]} (Zaid, {plan_families[2]})"
        )

        return {"rotation": rotation, "summary": summary}

    # ──────────────────────────────────────────────────────────────
    # Live weather auto-fetch (Open-Meteo — free, no API key)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def fetch_weather(lat: float, lon: float, timeout: int = 10) -> dict:
        """Fetch current weather from Open-Meteo API for a given location.

        Returns a dict with keys matching ``predict()`` weather params:
            weather_temp, humidity, rainfall, sunshine, wind_speed.

        Uses the *current_weather* + *daily* endpoints for a 7-day average
        of rainfall and sunshine so the values better represent seasonal
        conditions rather than a single-day snapshot.

        Parameters
        ----------
        lat, lon : float
            GPS coordinates.
        timeout : int
            HTTP request timeout in seconds.

        Returns
        -------
        dict  with weather_temp, humidity, rainfall, sunshine, wind_speed.

        Raises
        ------
        RuntimeError  if the API call fails.
        """
        import urllib.request
        import urllib.error

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"precipitation"
            f"&daily=precipitation_sum,sunshine_duration"
            f"&timezone=Asia%2FKolkata&forecast_days=7"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CropRecommender/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                import json as _json
                data = _json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Open-Meteo API request failed: {exc}") from exc

        current = data.get("current", {})
        daily = data.get("daily", {})

        # 7-day average rainfall (mm/day → seasonal proxy)
        precip_daily = daily.get("precipitation_sum", [0])
        avg_rain_per_day = sum(precip_daily) / max(len(precip_daily), 1)
        # Estimate seasonal rainfall (rough 120-day season)
        seasonal_rainfall = round(avg_rain_per_day * 120, 1)

        # 7-day average sunshine (seconds → hours)
        sunshine_secs = daily.get("sunshine_duration", [0])
        avg_sunshine_hrs = round(
            sum(sunshine_secs) / max(len(sunshine_secs), 1) / 3600, 1
        )

        weather = {
            "weather_temp": current.get("temperature_2m", 25.0),
            "humidity": current.get("relative_humidity_2m", 60),
            "rainfall": seasonal_rainfall,
            "sunshine": avg_sunshine_hrs,
            "wind_speed": current.get("wind_speed_10m", 8.0),
        }
        logger.info(
            "Fetched weather for (%.2f, %.2f): temp=%.1f°C hum=%d%% "
            "rain=%.0fmm sun=%.1fh wind=%.1fkm/h",
            lat, lon, weather["weather_temp"], weather["humidity"],
            weather["rainfall"], weather["sunshine"], weather["wind_speed"],
        )
        return weather

    def predict_with_live_weather(self, **kwargs) -> dict:
        """Like ``predict()`` but auto-fills weather fields from Open-Meteo.

        Any explicitly provided weather_temp / humidity / rainfall /
        sunshine / wind_speed values will override the fetched ones.
        """
        lat = kwargs["lat"]
        lon = kwargs["lon"]

        try:
            live = self.fetch_weather(lat, lon)
        except RuntimeError as exc:
            logger.warning("Weather fetch failed, using fallback defaults: %s", exc)
            live = self._SEASON_WEATHER.get(kwargs.get("season", "Kharif"), {})

        # User-provided values take precedence
        for key in ("weather_temp", "humidity", "rainfall", "sunshine", "wind_speed"):
            if key not in kwargs:
                kwargs[key] = live.get(key, self._SEASON_WEATHER["Kharif"].get(key, 0))

        return self.predict(**kwargs)

    # ──────────────────────────────────────────────────────────────
    # Soil amendment / fertilizer gap calculator
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_amendments(
        crop_name: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        field_area_ha: float = 1.0,
    ) -> dict:
        """Calculate NPK fertilizer gap for a target crop.

        Compares current soil NPK (mg/kg) against the crop's ideal midpoint
        from crop_profiles.py and converts the deficit to kg/ha of common
        fertilizers (Urea, DAP, MOP).

        Parameters
        ----------
        crop_name : str
            Target crop, e.g. "Soybean", "Wheat".
        nitrogen, phosphorus, potassium : float
            Current soil NPK in mg/kg.
        field_area_ha : float
            Field area in hectares (default 1.0).

        Returns
        -------
        dict with:
            crop: str,
            current_npk: dict, ideal_npk: dict, gap_npk: dict,
            fertilizer_kg_per_ha: dict (Urea, DAP, MOP),
            total_for_field: dict (scaled to field_area_ha),
            notes: list[str]
        """
        from generators.crop_profiles import ALL_CROPS

        # Find the crop profile
        profile = None
        for _season, crops in ALL_CROPS.items():
            if crop_name in crops:
                profile = crops[crop_name]
                break
        if profile is None:
            return {"error": f"Crop '{crop_name}' not found in profiles."}

        # Ideal midpoints
        ideal_n = (profile["n_range"][0] + profile["n_range"][1]) / 2
        ideal_p = (profile["p_range"][0] + profile["p_range"][1]) / 2
        ideal_k = (profile["k_range"][0] + profile["k_range"][1]) / 2

        # Gaps (only deficit, no excess removal)
        gap_n = max(0, ideal_n - nitrogen)
        gap_p = max(0, ideal_p - phosphorus)
        gap_k = max(0, ideal_k - potassium)

        # Convert mg/kg gap to kg/ha (assuming 2M kg soil per ha in top 20cm)
        # Then to common fertilizers:
        #   Urea = 46% N  →  kg_urea = gap_n_kg / 0.46
        #   DAP  = 46% P2O5 (≈20% P)  →  kg_dap = gap_p_kg / 0.20
        #   MOP  = 60% K2O (≈50% K)   →  kg_mop = gap_k_kg / 0.50
        soil_mass_kg_per_ha = 2_000_000  # top 20 cm
        gap_n_kg = gap_n * soil_mass_kg_per_ha / 1_000_000
        gap_p_kg = gap_p * soil_mass_kg_per_ha / 1_000_000
        gap_k_kg = gap_k * soil_mass_kg_per_ha / 1_000_000

        urea_kg = round(gap_n_kg / 0.46, 1)
        dap_kg = round(gap_p_kg / 0.20, 1)
        mop_kg = round(gap_k_kg / 0.50, 1)

        notes = []
        if gap_n == 0 and gap_p == 0 and gap_k == 0:
            notes.append("Soil NPK meets or exceeds crop requirements — no amendments needed.")
        else:
            if gap_n > 0:
                notes.append(f"Nitrogen deficit: {gap_n:.0f} mg/kg → apply ~{urea_kg} kg Urea/ha")
            if gap_p > 0:
                notes.append(f"Phosphorus deficit: {gap_p:.0f} mg/kg → apply ~{dap_kg} kg DAP/ha")
            if gap_k > 0:
                notes.append(f"Potassium deficit: {gap_k:.0f} mg/kg → apply ~{mop_kg} kg MOP/ha")

        return {
            "crop": crop_name,
            "current_npk": {"N": nitrogen, "P": phosphorus, "K": potassium},
            "ideal_npk": {"N": round(ideal_n, 1), "P": round(ideal_p, 1), "K": round(ideal_k, 1)},
            "gap_npk": {"N": round(gap_n, 1), "P": round(gap_p, 1), "K": round(gap_k, 1)},
            "fertilizer_kg_per_ha": {"Urea": urea_kg, "DAP": dap_kg, "MOP": mop_kg},
            "total_for_field": {
                "Urea_kg": round(urea_kg * field_area_ha, 1),
                "DAP_kg": round(dap_kg * field_area_ha, 1),
                "MOP_kg": round(mop_kg * field_area_ha, 1),
                "field_area_ha": field_area_ha,
            },
            "notes": notes,
        }


def demo():
    """Run a quick demo with sample inputs."""
    import sys
    import traceback
    outfile = open(BASE_DIR / "inference_demo_results.txt", "w", encoding="utf-8")
    _orig = sys.stdout
    sys.stdout = outfile

    try:
        recommender = CropRecommender()
        print("=" * 80)
        print("CROP RECOMMENDATION ENGINE - Top-3 Inference Demo")
        print("=" * 80)

        test_cases = [
            {"name": "Soybean scenario (Vidarbha Kharif)",
             "args": dict(nitrogen=110, phosphorus=55, potassium=70,
                          temperature=26, moisture=55, ec=1400, ph=6.0,
                          weather_temp=27, humidity=75, rainfall=900,
                          sunshine=4.5, wind_speed=8,
                          lat=20.93, lon=77.75, altitude=343,
                          organic_carbon=0.67,
                          soil_type="Black (Regur)", soil_texture="Clay Loam",
                          drainage="Moderate", agro_zone="Vidarbha",
                          season="Kharif", month=7)},
            {"name": "Onion scenario (Nashik Rabi)",
             "args": dict(nitrogen=125, phosphorus=100, potassium=150,
                          temperature=19, moisture=30, ec=850, ph=6.8,
                          weather_temp=20, humidity=50, rainfall=450,
                          sunshine=7.5, wind_speed=6,
                          lat=20.0, lon=74.0, altitude=580,
                          organic_carbon=0.4,
                          soil_type="Red", soil_texture="Sandy Loam",
                          drainage="Excessive", agro_zone="North Maharashtra",
                          season="Rabi", month=12)},
            {"name": "Out-of-distribution (UP Wheat)",
             "args": dict(nitrogen=150, phosphorus=100, potassium=150,
                          temperature=15, moisture=45, ec=1100, ph=7.0,
                          weather_temp=16, humidity=55, rainfall=400,
                          sunshine=7, wind_speed=5,
                          lat=26.85, lon=80.91, altitude=98,
                          organic_carbon=0.6,
                          soil_type="Alluvial", soil_texture="Silt Loam",
                          drainage="Good", agro_zone="Indo-Gangetic Plain",
                          season="Rabi", month=12)},
        ]

        for tc in test_cases:
            print(f"\n{'-' * 80}")
            print(f"SCENARIO: {tc['name']}")
            print(f"{'-' * 80}")
            result = recommender.predict(**tc["args"])
            print(f"\n  Overall Flag: {result['confidence_flag']}")
            print(f"  OOD:          {result['is_ood']}")
            print(f"\n  Top-3 Recommendations:")
            for i, crop in enumerate(result["top_3"], 1):
                print(f"    {i}. {crop['crop']:20s}  {crop['confidence_pct']:>6s}  [{crop['flag']}]")
            print(f"\n  Advisory: {result['advisory']}")

        print(f"\n{'=' * 80}")
        print("DEMO COMPLETE")
        print(f"{'=' * 80}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = _orig
        outfile.close()
    print(f"Results written to {BASE_DIR / 'inference_demo_results.txt'}")


if __name__ == "__main__":
    demo()

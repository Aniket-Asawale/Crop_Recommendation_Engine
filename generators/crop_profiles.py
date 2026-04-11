"""
Crop Profiles — Agronomic requirements for 22+ crops across Kharif, Rabi, Zaid seasons.
Used by sensor_data_generator.py and crop_label_validator.py.

Each crop defines: soil affinities, pH range, NPK requirements, rainfall/temp needs.
Source: PLAN.md Section 5 + ICAR Maharashtra guidelines.
"""

# Crop family taxonomy
CROP_FAMILIES = {
    "Cereal":    ["Rice", "Wheat", "Jowar (Kharif)", "Rabi Jowar", "Bajra", "Maize"],
    "Legume":    ["Soybean", "Chickpea (Gram)", "Pigeonpea (Tur)", "Green Gram", "Black Gram", "Lentil"],
    "Oilseed":   ["Groundnut", "Sunflower", "Linseed", "Safflower", "Sesame"],
    "Cash":      ["Cotton", "Sugarcane", "Grape", "Onion", "Turmeric"],
    "Vegetable": ["Tomato", "Chilli", "Okra", "Brinjal"],
}

# Reverse lookup: crop_name -> crop_family
CROP_TO_FAMILY = {}
for family, crops in CROP_FAMILIES.items():
    for crop in crops:
        CROP_TO_FAMILY[crop] = family


# ─── KHARIF CROPS (June – October) ───
KHARIF_CROPS = {
    "Soybean": {
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (5.5, 7.5), "n_range": (80, 140), "p_range": (40, 70), "k_range": (50, 90),
        "rainfall_mm": (700, 1100), "temp_range": (20, 30),
    },
    "Cotton": {
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (6.5, 8.0), "n_range": (100, 200), "p_range": (50, 90), "k_range": (120, 200),
        "rainfall_mm": (500, 800), "temp_range": (21, 35),
    },
    "Jowar (Kharif)": {
        "soil_affinity": ["Black (Regur)", "Red"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (6.0, 8.0), "n_range": (100, 150), "p_range": (30, 60), "k_range": (80, 130),
        "rainfall_mm": (400, 700), "temp_range": (25, 35),
    },
    "Bajra": {
        "soil_affinity": ["Sandy", "Red"],
        "soil_secondary": ["Alluvial", "Laterite"],
        "ph_range": (5.5, 7.5), "n_range": (80, 120), "p_range": (20, 50), "k_range": (60, 100),
        "rainfall_mm": (300, 600), "temp_range": (25, 35),
    },
    "Maize": {
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (5.8, 7.5), "n_range": (150, 200), "p_range": (60, 90), "k_range": (140, 180),
        "rainfall_mm": (500, 900), "temp_range": (20, 32),
    },
    "Rice": {
        "soil_affinity": ["Alluvial", "Clay"],
        "soil_secondary": ["Laterite", "Black (Regur)"],
        "ph_range": (5.5, 7.0), "n_range": (150, 200), "p_range": (60, 80), "k_range": (100, 150),
        "rainfall_mm": (1000, 2000), "temp_range": (22, 35),
    },
    "Groundnut": {
        "soil_affinity": ["Red", "Sandy"],
        "soil_secondary": ["Alluvial", "Laterite"],
        "ph_range": (5.5, 7.0), "n_range": (40, 80), "p_range": (40, 70), "k_range": (80, 130),
        "rainfall_mm": (500, 800), "temp_range": (22, 33),
    },
    "Sugarcane": {
        "soil_affinity": ["Alluvial", "Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (6.0, 8.0), "n_range": (120, 300), "p_range": (60, 150), "k_range": (80, 300),
        "rainfall_mm": (1500, 2500), "temp_range": (20, 38),
    },
    "Pigeonpea (Tur)": {
        "soil_affinity": ["Black (Regur)", "Red"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (6.8, 7.8), "n_range": (35, 65), "p_range": (50, 75), "k_range": (110, 150),
        "rainfall_mm": (500, 850), "temp_range": (25, 35),
    },
    "Green Gram": {
        "soil_affinity": ["Sandy", "Laterite"],
        "soil_secondary": ["Red"],
        "ph_range": (5.5, 6.8), "n_range": (20, 45), "p_range": (25, 45), "k_range": (40, 70),
        "rainfall_mm": (350, 600), "temp_range": (28, 38),
    },
    "Sesame": {
        "soil_affinity": ["Sandy", "Red"],
        "soil_secondary": ["Alluvial", "Laterite"],
        "ph_range": (5.5, 7.5), "n_range": (40, 80), "p_range": (30, 50), "k_range": (40, 80),
        "rainfall_mm": (300, 600), "temp_range": (25, 38),
    },
    "Turmeric": {
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Laterite"],
        "ph_range": (5.5, 7.0), "n_range": (100, 160), "p_range": (60, 100), "k_range": (120, 180),
        "rainfall_mm": (800, 1500), "temp_range": (20, 35),
    },
    "Black Gram": {
        "soil_affinity": ["Alluvial", "Laterite"],
        "soil_secondary": ["Red", "Sandy"],
        "ph_range": (6.5, 7.5), "n_range": (15, 40), "p_range": (20, 38), "k_range": (35, 60),
        "rainfall_mm": (450, 750), "temp_range": (27, 36),
    },
}


# ─── RABI CROPS (November – February) ───
RABI_CROPS = {
    "Wheat": {
        "soil_affinity": ["Alluvial"],
        "soil_secondary": ["Black (Regur)", "Clay"],
        "ph_range": (6.0, 7.5), "n_range": (120, 180), "p_range": (80, 120), "k_range": (120, 180),
        "rainfall_mm": (250, 600), "temp_range": (10, 25),
    },
    "Chickpea (Gram)": {
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Red", "Alluvial"],
        "ph_range": (6.0, 8.0), "n_range": (20, 40), "p_range": (40, 60), "k_range": (80, 120),
        "rainfall_mm": (200, 500), "temp_range": (10, 25),
    },
    "Rabi Jowar": {
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (6.5, 8.0), "n_range": (80, 130), "p_range": (30, 60), "k_range": (80, 130),
        "rainfall_mm": (200, 500), "temp_range": (12, 30),
    },
    "Sunflower": {
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.0, 7.5), "n_range": (80, 120), "p_range": (60, 90), "k_range": (100, 150),
        "rainfall_mm": (250, 500), "temp_range": (15, 30),
    },
    "Linseed": {
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (5.5, 7.0), "n_range": (60, 100), "p_range": (30, 50), "k_range": (60, 100),
        "rainfall_mm": (200, 450), "temp_range": (10, 25),
    },
    "Safflower": {
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (6.5, 8.0), "n_range": (60, 90), "p_range": (40, 60), "k_range": (80, 130),
        "rainfall_mm": (200, 450), "temp_range": (15, 30),
    },
    "Onion": {
        "soil_affinity": ["Red", "Sandy"],
        "soil_secondary": ["Alluvial", "Black (Regur)"],
        "ph_range": (6.0, 7.5), "n_range": (100, 150), "p_range": (80, 120), "k_range": (120, 180),
        "rainfall_mm": (300, 600), "temp_range": (13, 25),
    },
    "Grape": {
        "soil_affinity": ["Red", "Laterite"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.0, 7.5), "n_range": (100, 140), "p_range": (80, 120), "k_range": (180, 250),
        "rainfall_mm": (250, 600), "temp_range": (15, 35),
    },
    "Lentil": {
        "soil_affinity": ["Alluvial", "Red"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.0, 7.5), "n_range": (20, 40), "p_range": (30, 50), "k_range": (60, 100),
        "rainfall_mm": (200, 450), "temp_range": (10, 25),
    },
}


# ─── ZAID CROPS (March – May) ───
# NOTE: Profiles deliberately spaced for ML separability while remaining agronomically valid.
ZAID_CROPS = {
    "Okra": {
        "soil_affinity": ["Sandy", "Laterite"],
        "soil_secondary": ["Red"],
        "ph_range": (5.5, 6.5), "n_range": (35, 65), "p_range": (15, 35), "k_range": (25, 55),
        "rainfall_mm": (50, 200), "temp_range": (32, 42),
    },
    "Tomato": {
        "soil_affinity": ["Red", "Laterite"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (5.0, 6.2), "n_range": (150, 220), "p_range": (90, 130), "k_range": (160, 240),
        "rainfall_mm": (200, 450), "temp_range": (18, 28),
    },
    "Chilli": {
        "soil_affinity": ["Red", "Sandy"],
        "soil_secondary": ["Laterite", "Alluvial"],
        "ph_range": (6.0, 7.5), "n_range": (50, 90), "p_range": (25, 50), "k_range": (40, 80),
        "rainfall_mm": (50, 180), "temp_range": (30, 40),
    },
    "Brinjal": {
        "soil_affinity": ["Alluvial", "Clay"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.2, 7.2), "n_range": (110, 160), "p_range": (65, 95), "k_range": (110, 170),
        "rainfall_mm": (250, 500), "temp_range": (22, 32),
    },
}


# ─── Combined lookup ───
ALL_CROPS = {
    "Kharif": KHARIF_CROPS,
    "Rabi": RABI_CROPS,
    "Zaid": ZAID_CROPS,
}

# Reverse lookup: crop name → season (for hard season filtering in inference)
CROP_TO_SEASON = {}
for _season, _crops in ALL_CROPS.items():
    for _crop_name in _crops:
        CROP_TO_SEASON[_crop_name] = _season

# Season month mappings
SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9, 10],
    "Rabi": [11, 12, 1, 2],
    "Zaid": [3, 4, 5],
}

# Soil profiles from AgroSensor (for NPK baselines and sensor generation)
SOIL_PROFILES = {
    "Alluvial":      {"water_retention": 0.65, "drainage_rate": 0.6,  "ph_base": 7.0, "ph_range": (6.5, 8.0), "ec_base": 1100, "ec_range": (400, 2200),  "n_base": 140, "p_base": 90, "k_base": 170},
    "Black (Regur)": {"water_retention": 0.85, "drainage_rate": 0.3,  "ph_base": 7.8, "ph_range": (7.0, 8.5), "ec_base": 1400, "ec_range": (600, 2800),  "n_base": 110, "p_base": 70, "k_base": 200},
    "Red":           {"water_retention": 0.40, "drainage_rate": 0.75, "ph_base": 5.8, "ph_range": (4.5, 6.5), "ec_base": 800,  "ec_range": (200, 1500),  "n_base": 80,  "p_base": 50, "k_base": 100},
    "Laterite":      {"water_retention": 0.35, "drainage_rate": 0.80, "ph_base": 5.5, "ph_range": (4.5, 6.0), "ec_base": 600,  "ec_range": (150, 1200),  "n_base": 60,  "p_base": 40, "k_base": 80},
    "Sandy":         {"water_retention": 0.20, "drainage_rate": 0.90, "ph_base": 6.5, "ph_range": (5.5, 7.5), "ec_base": 500,  "ec_range": (100, 1000),  "n_base": 50,  "p_base": 30, "k_base": 60},
    "Clay":          {"water_retention": 0.90, "drainage_rate": 0.20, "ph_base": 7.2, "ph_range": (6.5, 8.5), "ec_base": 1500, "ec_range": (700, 3000),  "n_base": 130, "p_base": 80, "k_base": 180},
}


# ─── Regional Crop Dominance (agro_zone → season → crops with boost weights) ───
# Based on Maharashtra State Agriculture Census cropping area data.
# These biases ensure the generator produces region-realistic crop distributions.
REGIONAL_CROP_DOMINANCE = {
    "Vidarbha": {
        "Kharif": {"Cotton": 2.5, "Soybean": 2.5, "Pigeonpea (Tur)": 2.0, "Jowar (Kharif)": 1.3, "Green Gram": 1.2},
        "Rabi":   {"Chickpea (Gram)": 2.5, "Wheat": 1.8, "Rabi Jowar": 1.5, "Safflower": 1.5, "Linseed": 1.3},
        "Zaid":   {"Okra": 1.2, "Chilli": 1.0},
    },
    "Marathwada": {
        "Kharif": {"Cotton": 2.0, "Soybean": 2.0, "Bajra": 1.5, "Pigeonpea (Tur)": 1.8, "Jowar (Kharif)": 1.5},
        "Rabi":   {"Rabi Jowar": 2.5, "Chickpea (Gram)": 2.0, "Sunflower": 1.5, "Safflower": 1.5},
        "Zaid":   {"Okra": 1.0, "Chilli": 1.0},
    },
    "Western Maharashtra": {
        "Kharif": {"Sugarcane": 2.5, "Rice": 1.8, "Soybean": 1.5, "Groundnut": 1.3, "Maize": 1.3},
        "Rabi":   {"Wheat": 2.0, "Onion": 2.5, "Grape": 2.0, "Chickpea (Gram)": 1.5},
        "Zaid":   {"Tomato": 1.5, "Okra": 1.2, "Brinjal": 1.2},
    },
    "Konkan": {
        "Kharif": {"Rice": 3.0, "Groundnut": 1.5, "Sesame": 1.3, "Black Gram": 1.3},
        "Rabi":   {"Lentil": 1.5, "Chickpea (Gram)": 1.0},
        "Zaid":   {"Okra": 1.3, "Brinjal": 1.2},
    },
    "North Maharashtra": {
        "Kharif": {"Bajra": 2.0, "Maize": 1.8, "Cotton": 1.5, "Groundnut": 1.5, "Sugarcane": 1.3},
        "Rabi":   {"Onion": 2.5, "Wheat": 2.0, "Grape": 2.0, "Chickpea (Gram)": 1.5},
        "Zaid":   {"Tomato": 1.5, "Chilli": 1.2, "Okra": 1.0},
    },
}


# ─── EC-sensitive crops (penalised when EC > threshold) ───
# Crops that do poorly in saline/alkaline soils with high electrical conductivity.
EC_SENSITIVE_CROPS = {
    # crop_name: max_ec_threshold (μS/cm) — above this, crop is penalised
    "Soybean": 1600,
    "Green Gram": 1200,
    "Black Gram": 1400,
    "Sesame": 1200,
    "Lentil": 1200,
    "Okra": 1400,
    "Chilli": 1400,
    "Tomato": 1600,
    "Brinjal": 1800,
    "Groundnut": 1500,
    "Sunflower": 2000,
    "Maize": 1800,
}

# EC-tolerant crops (can handle higher EC)
EC_TOLERANT_CROPS = {
    "Cotton": 3500,
    "Sugarcane": 3000,
    "Wheat": 2800,
    "Bajra": 3000,
    "Rabi Jowar": 2800,
    "Jowar (Kharif)": 2800,
    "Safflower": 2500,
    "Rice": 2500,
}


# ─── Soil-Crop Incompatibility (hard penalties for the guardrail layer) ───
# Crops that should NEVER be recommended for certain soil types.
SOIL_CROP_INCOMPATIBLE = {
    "Black (Regur)": ["Chilli", "Okra", "Groundnut", "Sesame"],       # heavy clay — bad for light-soil crops
    "Clay":          ["Chilli", "Okra", "Groundnut", "Sesame"],
    "Sandy":         ["Rice", "Sugarcane"],                            # can't retain water
    "Laterite":      ["Rice", "Sugarcane", "Cotton"],                  # poor fertility
}


# ─── Drainage-Crop Compatibility ───
# Crops that THRIVE under poor drainage (waterlogged / standing water)
DRAINAGE_TOLERANT_CROPS = {"Rice", "Sugarcane"}

# Crops that FAIL under poor drainage (need well-drained soil)
DRAINAGE_SENSITIVE_CROPS = {
    "Chilli", "Brinjal", "Tomato", "Okra",        # vegetables — root rot
    "Onion",                                       # bulb rot in waterlogging
    "Groundnut", "Sesame",                         # oilseeds — need drainage
    "Soybean",                                     # moderate tolerance but not poor
    "Cotton",                                      # boll rot in standing water
    "Sunflower", "Safflower",                      # oilseeds
    "Jowar (Kharif)", "Rabi Jowar", "Bajra",      # dryland cereals — waterlogging intolerant
    "Maize",                                       # root suffocation in standing water
}

# Valid drainage values per soil type (for input consistency validation)
# Based on SOIL_PROPERTIES from location_generator.py
SOIL_DRAINAGE_VALID = {
    "Sandy":         ["Good", "Excessive"],           # Sandy cannot have Poor drainage
    "Laterite":      ["Good", "Excessive"],
    "Red":           ["Good", "Moderate"],
    "Alluvial":      ["Good", "Moderate"],
    "Black (Regur)": ["Poor", "Moderate"],            # heavy clay → poor or moderate
    "Clay":          ["Very Poor", "Poor", "Moderate"],
}



# ─── Agro Zone Geographic Bounding Boxes ───
# Approximate lat/lon ranges for each Maharashtra agro zone.
# Used for input validation: flagging zone-location mismatches.
AGRO_ZONE_BOUNDS = {
    "Vidarbha":              {"lat": (19.5, 22.0), "lon": (76.0, 80.5)},
    "Marathwada":            {"lat": (17.5, 20.5), "lon": (74.5, 77.5)},
    "Western Maharashtra":   {"lat": (15.5, 19.5), "lon": (73.5, 76.0)},
    "Konkan":                {"lat": (15.5, 20.0), "lon": (72.5, 74.0)},
    "North Maharashtra":     {"lat": (19.5, 22.0), "lon": (72.5, 76.0)},
    "Northern Maharashtra":  {"lat": (19.5, 22.0), "lon": (72.5, 76.0)},
}
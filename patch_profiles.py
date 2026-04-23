import os

file_path = r"c:\Users\Aniket\OneDrive\Documents\College\SEM7\project\AgroModules\Crop_Recommendation_Engine\generators\crop_profiles.py"

with open(file_path, "a", encoding="utf-8") as f:
    f.write("""

# ─── Crop Decision Engine Constants (T8) ───

CROP_YIELD_BENCHMARKS = {
    # Crop: max_t_ha, avg_t_ha, unit, duration_days
    "Cotton": {"max_t_ha": 3.0, "avg_t_ha": 1.2, "unit": "t lint/ha", "duration_days": 150},
    "Soybean": {"max_t_ha": 3.5, "avg_t_ha": 1.5, "unit": "t/ha", "duration_days": 100},
    "Wheat": {"max_t_ha": 5.5, "avg_t_ha": 3.2, "unit": "t/ha", "duration_days": 120},
    "Rice": {"max_t_ha": 7.5, "avg_t_ha": 3.5, "unit": "t/ha", "duration_days": 130},
    "Jowar (Kharif)": {"max_t_ha": 4.0, "avg_t_ha": 1.5, "unit": "t/ha", "duration_days": 110},
    "Rabi Jowar": {"max_t_ha": 3.5, "avg_t_ha": 1.2, "unit": "t/ha", "duration_days": 110},
    "Bajra": {"max_t_ha": 3.0, "avg_t_ha": 1.0, "unit": "t/ha", "duration_days": 90},
    "Maize": {"max_t_ha": 8.0, "avg_t_ha": 3.5, "unit": "t/ha", "duration_days": 110},
    "Ragi": {"max_t_ha": 3.5, "avg_t_ha": 1.5, "unit": "t/ha", "duration_days": 120},
    "Chickpea (Gram)": {"max_t_ha": 2.5, "avg_t_ha": 1.1, "unit": "t/ha", "duration_days": 110},
    "Pigeonpea (Tur)": {"max_t_ha": 2.5, "avg_t_ha": 1.0, "unit": "t/ha", "duration_days": 160},
    "Green Gram": {"max_t_ha": 1.5, "avg_t_ha": 0.6, "unit": "t/ha", "duration_days": 65},
    "Black Gram": {"max_t_ha": 1.5, "avg_t_ha": 0.6, "unit": "t/ha", "duration_days": 70},
    "Lentil": {"max_t_ha": 2.0, "avg_t_ha": 0.9, "unit": "t/ha", "duration_days": 110},
    "Groundnut": {"max_t_ha": 3.5, "avg_t_ha": 1.6, "unit": "t/ha", "duration_days": 110},
    "Sunflower": {"max_t_ha": 2.5, "avg_t_ha": 0.9, "unit": "t/ha", "duration_days": 100},
    "Linseed": {"max_t_ha": 1.5, "avg_t_ha": 0.5, "unit": "t/ha", "duration_days": 120},
    "Safflower": {"max_t_ha": 2.0, "avg_t_ha": 0.8, "unit": "t/ha", "duration_days": 130},
    "Sesame": {"max_t_ha": 1.2, "avg_t_ha": 0.4, "unit": "t/ha", "duration_days": 90},
    "Mustard": {"max_t_ha": 2.5, "avg_t_ha": 1.2, "unit": "t/ha", "duration_days": 110},
    "Sugarcane": {"max_t_ha": 120.0, "avg_t_ha": 80.0, "unit": "t/ha", "duration_days": 360},
    "Grape": {"max_t_ha": 35.0, "avg_t_ha": 22.0, "unit": "t/ha", "duration_days": 180},
    "Onion": {"max_t_ha": 30.0, "avg_t_ha": 16.0, "unit": "t/ha", "duration_days": 120},
    "Turmeric": {"max_t_ha": 25.0, "avg_t_ha": 12.0, "unit": "t/ha", "duration_days": 240},
    "Tomato": {"max_t_ha": 60.0, "avg_t_ha": 25.0, "unit": "t/ha", "duration_days": 130},
    "Chilli": {"max_t_ha": 15.0, "avg_t_ha": 6.0, "unit": "t/ha (dry)", "duration_days": 150},
    "Okra": {"max_t_ha": 18.0, "avg_t_ha": 10.0, "unit": "t/ha", "duration_days": 100},
    "Brinjal": {"max_t_ha": 45.0, "avg_t_ha": 20.0, "unit": "t/ha", "duration_days": 140},
    "Potato": {"max_t_ha": 40.0, "avg_t_ha": 22.0, "unit": "t/ha", "duration_days": 110},
    "Coriander": {"max_t_ha": 1.5, "avg_t_ha": 0.7, "unit": "t/ha (seed)", "duration_days": 100},
    "Banana": {"max_t_ha": 70.0, "avg_t_ha": 40.0, "unit": "t/ha", "duration_days": 300},
    "Mango": {"max_t_ha": 12.0, "avg_t_ha": 6.0, "unit": "t/ha", "duration_days": 365},
    "Pomegranate": {"max_t_ha": 20.0, "avg_t_ha": 12.0, "unit": "t/ha", "duration_days": 300},
    "Cashew": {"max_t_ha": 2.5, "avg_t_ha": 1.0, "unit": "t/ha", "duration_days": 365},
    "Coconut": {"max_t_ha": 15000, "avg_t_ha": 8000, "unit": "nuts/ha", "duration_days": 365},
}

CROP_MSP = {
    # Approx MSP or market price per tonne in INR (2025/2026 estimates)
    "Cotton": 75000,
    "Soybean": 48000,
    "Wheat": 22750,
    "Rice": 23000,
    "Jowar (Kharif)": 31800,
    "Rabi Jowar": 32250,
    "Bajra": 25000,
    "Maize": 20900,
    "Ragi": 38460,
    "Chickpea (Gram)": 54400,
    "Pigeonpea (Tur)": 70000,
    "Green Gram": 85580,
    "Black Gram": 69500,
    "Lentil": 64250,
    "Groundnut": 63770,
    "Sunflower": 67600,
    "Linseed": 58000,
    "Safflower": 58000,
    "Sesame": 86350,
    "Mustard": 56500,
    "Sugarcane": 3400, # Per tonne
    "Grape": 40000, # Market price
    "Onion": 15000, # Market price
    "Turmeric": 140000,
    "Tomato": 12000,
    "Chilli": 180000, # Dry
    "Okra": 20000,
    "Brinjal": 15000,
    "Potato": 12000,
    "Coriander": 85000,
    "Banana": 15000,
    "Mango": 60000,
    "Pomegranate": 80000,
    "Cashew": 100000,
    "Coconut": 15, # Per nut
}

CROP_AGRONOMY = {
    "Cotton": {
        "sow_months": [6, 7],
        "harvest_months": [11, 12, 1],
        "seed_rate": "2.5 kg/ha (Bt hybrids)",
        "spacing": "90×30 cm (120×45 for drip)",
        "irrigation_count": "6–8",
        "irrigation_mm": 50,
        "pest_watch": ["Bollworm", "Jassids", "Whiteflies", "Aphids"],
        "fert_splits": ["25% basal", "50% at square formation", "25% at boll formation"],
        "next_crop": "Chickpea or Wheat",
    },
    "Soybean": {
        "sow_months": [6, 7],
        "harvest_months": [9, 10],
        "seed_rate": "65-70 kg/ha",
        "spacing": "45×15 cm",
        "irrigation_count": "Rainfed (1-2 protective if dry)",
        "irrigation_mm": 50,
        "pest_watch": ["Girdle beetle", "Stem fly", "Defoliators"],
        "fert_splits": ["100% basal (N-P-K) + Sulphur"],
        "next_crop": "Wheat or Chickpea",
    },
    "Wheat": {
        "sow_months": [11, 12],
        "harvest_months": [3, 4],
        "seed_rate": "100-125 kg/ha",
        "spacing": "22.5 cm (row)",
        "irrigation_count": "4–6",
        "irrigation_mm": 60,
        "pest_watch": ["Rusts (Brown/Yellow)", "Aphids", "Termites"],
        "fert_splits": ["50% N + 100% P&K basal", "50% N at Crown Root Initiation (CRI)"],
        "next_crop": "Green gram or Fallow",
    },
    "Rice": {
        "sow_months": [6, 7],
        "harvest_months": [10, 11],
        "seed_rate": "40-50 kg/ha (transplanted)",
        "spacing": "20×15 cm",
        "irrigation_count": "Continuous flooding / AWD",
        "irrigation_mm": "Maintain 5cm standing water",
        "pest_watch": ["Stem borer", "Leaf folder", "Blast", "BPH"],
        "fert_splits": ["50% N, 100% P&K basal", "25% N tillering", "25% N panicle initiation"],
        "next_crop": "Gram or Lentil",
    },
    "Potato": {
        "sow_months": [10, 11],
        "harvest_months": [2, 3],
        "seed_rate": "2.5-3.0 t/ha (tubers)",
        "spacing": "60×20 cm",
        "irrigation_count": "7–10 (light & frequent)",
        "irrigation_mm": 40,
        "pest_watch": ["Late Blight", "Aphids", "White grubs"],
        "fert_splits": ["50% N, 100% P&K basal", "50% N at earthing up (30 DAS)"],
        "next_crop": "Zaid vegetables or Green gram",
    },
    "Coconut": {
        "sow_months": [6, 7, 8],
        "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "seed_rate": "140-150 seedlings/ha",
        "spacing": "7.5×7.5 m",
        "irrigation_count": "Summer months (Drip preferred)",
        "irrigation_mm": "50-60 liters/palm/day in summer",
        "pest_watch": ["Rhinoceros beetle", "Red palm weevil", "Bud rot"],
        "fert_splits": ["1/3rd after monsoon", "2/3rd before post-monsoon"],
        "next_crop": "Intercrops (Banana, Spices)",
    }
}
""")

"""Cross-validation: 6 ChatGPT-provided test cases."""
import requests, json

BASE = "http://127.0.0.1:8002"

# Common weather defaults (API requires these)
WX = dict(weather_temp=27.0, humidity=65.0, rainfall=120.0, sunshine=7.0, wind_speed=10.0)

CASES = [
    ("1) Vidarbha Soybean (ideal)", "Soybean", dict(
        nitrogen=90, phosphorus=45, potassium=50, temperature=25.5, moisture=60.0,
        ec=900, ph=6.8, lat=20.5, lon=78.9, altitude=320, organic_carbon=0.7,
        soil_type="Black (Regur)", soil_texture="Clay Loam", drainage="Moderate",
        agro_zone="Vidarbha", season="Kharif", month=7, **WX)),

    ("2) Vidarbha Cotton (high EC)", "Cotton", dict(
        nitrogen=110, phosphorus=60, potassium=80, temperature=27.0, moisture=55.0,
        ec=1400, ph=7.5, lat=19.7, lon=77.5, altitude=350, organic_carbon=0.8,
        soil_type="Black (Regur)", soil_texture="Clay", drainage="Moderate",
        agro_zone="Vidarbha", season="Kharif", month=7, **WX)),

    ("3) Western MH Sugarcane (irrigated)", "Sugarcane", dict(
        nitrogen=140, phosphorus=70, potassium=90, temperature=26.0, moisture=70.0,
        ec=800, ph=7.2, lat=16.8, lon=74.5, altitude=550, organic_carbon=1.2,
        soil_type="Black (Regur)", soil_texture="Clay Loam", drainage="Moderate",
        agro_zone="Western Maharashtra", season="Kharif", month=7, **WX)),

    ("4) Konkan Rice (high rainfall)", "Rice", dict(
        nitrogen=80, phosphorus=40, potassium=40, temperature=26.5, moisture=85.0,
        ec=600, ph=6.2, lat=17.0, lon=73.3, altitude=100, organic_carbon=1.5,
        soil_type="Laterite", soil_texture="Clay", drainage="Poor",
        agro_zone="Konkan", season="Kharif", month=8,
        weather_temp=28.0, humidity=85.0, rainfall=350.0, sunshine=5.0, wind_speed=12.0)),

    ("5) Vidarbha Tur (rainfed, low moisture)", "Pigeonpea (Tur)", dict(
        nitrogen=60, phosphorus=30, potassium=40, temperature=28.0, moisture=40.0,
        ec=1000, ph=7.3, lat=19.8, lon=76.8, altitude=300, organic_carbon=0.6,
        soil_type="Black (Regur)", soil_texture="Clay Loam", drainage="Moderate",
        agro_zone="Vidarbha", season="Kharif", month=7, **WX)),

    ("6) INVALID (guardrail test)", "FLAGGED", dict(
        nitrogen=100, phosphorus=50, potassium=60, temperature=25.0, moisture=50.0,
        ec=900, ph=7.0, lat=19.2, lon=73.0, altitude=300, organic_carbon=1.0,
        soil_type="Laterite", soil_texture="Clay", drainage="Good",
        agro_zone="Vidarbha", season="Kharif", month=1, **WX)),
]

print("=" * 70)
print("CROSS-VALIDATION: 6 ChatGPT Test Cases")
print("=" * 70)

passed = 0
for name, expected, payload in CASES:
    print(f"\n--- {name} ---")
    print(f"  Expected: {expected}")
    try:
        r = requests.post(f"{BASE}/predict", json=payload, timeout=10)
        if r.status_code != 200:
            print(f"  ERROR {r.status_code}: {r.text[:200]}")
            continue
        d = r.json()
        top3 = d.get("top_3", [])
        for i, c in enumerate(top3):
            marker = " <<<" if c["crop"] == expected else ""
            print(f"  #{i+1} {c['crop']}: {c['confidence_pct']}{marker}")
            for n in c.get("guardrail_notes", []):
                print(f"      -> {n}")
        # Check advisory / warnings
        adv = d.get("advisory", "")
        if adv:
            print(f"  Advisory: {adv[:120]}")
        warns = d.get("input_warnings", [])
        if warns:
            for w in warns:
                print(f"  ⚠ {w}")

        if expected == "FLAGGED":
            if warns:
                print(f"  ✅ PASS — guardrails fired")
                passed += 1
            else:
                print(f"  ❌ FAIL — no warnings on invalid input")
        else:
            crops = [c["crop"] for c in top3]
            if expected in crops:
                rank = crops.index(expected) + 1
                print(f"  ✅ PASS — {expected} is #{rank}")
                passed += 1
            else:
                print(f"  ❌ FAIL — {expected} not in top 3: {crops}")
    except Exception as e:
        print(f"  CONNECTION ERROR: {e}")

print(f"\n{'=' * 70}")
print(f"RESULT: {passed}/{len(CASES)} cases passed")
print(f"{'=' * 70}")


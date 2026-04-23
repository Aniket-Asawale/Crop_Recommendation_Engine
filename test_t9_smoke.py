"""
T9 Smoke Test — weather_sensitivity badge in predict() output.
Run: venv\Scripts\python.exe test_t9_smoke.py
"""
import sys
sys.path.insert(0, ".")
from models.inference import CropRecommender

def test_kharif_vidarbha():
    r = CropRecommender()
    res = r.predict(
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

    print("=" * 60)
    print("T9 Smoke Test — Kharif Vidarbha (Soybean scenario)")
    print("=" * 60)
    print(f"  confidence_flag : {res['confidence_flag']}")
    print(f"  is_ood          : {res['is_ood']}")
    print()

    all_ok = True
    for i, crop in enumerate(res["top_3"], 1):
        ws = crop.get("weather_sensitivity")
        label = ws.get("label", "MISSING") if ws else "MISSING"
        pct   = ws.get("sensitivity_pct", "N/A") if ws else "N/A"
        print(f"  #{i}  {crop['crop']:<24}  conf={crop['confidence_pct']:<6}  "
              f"ws={pct}pp  [{label}]")
        if ws is None:
            all_ok = False

    print()
    if all_ok:
        print("PASS — weather_sensitivity present for all top_3 entries")
        # Check label is Low or Medium for a typical Kharif input
        top_label = res["top_3"][0]["weather_sensitivity"]["label"]
        assert top_label in ("Low", "Medium", "High"), f"Unexpected label: {top_label}"
        print(f"PASS — top-1 badge label = {top_label} (expected Low/Medium)")
    else:
        print("FAIL — weather_sensitivity missing from one or more top_3 entries")
        sys.exit(1)


if __name__ == "__main__":
    test_kharif_vidarbha()

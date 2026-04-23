import sys
sys.path.insert(0, ".")
from models.inference import CropRecommender
import json

def test():
    res = CropRecommender.evaluate_crop_decision(
        target_crop="Cotton",
        nitrogen=110, phosphorus=55, potassium=70,
        temperature=26, moisture=55, ec=1400, ph=6.0,
        weather_temp=27, humidity=75, rainfall=900,
        sunshine=4.5, wind_speed=8,
        lat=20.93, lon=77.75, altitude=343,
        organic_carbon=0.67,
        soil_type="Black (Regur)", soil_texture="Clay Loam",
        drainage="Moderate", agro_zone="Vidarbha",
        season="Kharif", month=7,
        field_area_ha=2.5
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test()

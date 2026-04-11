"""
Regenerate ALL batches with realistic seasonal weighting and merge.
Kharif ~50%, Rabi ~33%, Zaid ~17% (based on MH State Agriculture Census)

Usage: python Crop_Recommendation_Engine/generators/regenerate_all.py
"""

import sys
import json
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generators.sensor_data_generator import generate_batch
from generators.merge_batches import merge_and_validate

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
LOCATIONS_FILE = DATA_DIR / "locations_100.json"

# rows_per_season=4 with weights (3/2/1) gives:
#   Per location/year: Kharif=12, Rabi=8, Zaid=4 => 24 rows/year => 72 rows/location
#   Total: 100 locations * 72 = 7,200 rows
#   Kharif: 3600 (50%), Rabi: 2400 (33%), Zaid: 1200 (17%)
ROWS_PER_SEASON_BASE = 4

REGION_BATCHES = [
    ("Vidarbha", "Vidarbha"),
    ("Marathwada", "Marathwada"),
    ("Western Maharashtra", "Western_Maharashtra"),
    ("Konkan", "Konkan"),
    ("North Maharashtra", "North_Maharashtra"),
]


def main():
    with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
        all_locs = json.load(f)

    # Maharashtra batches by agro_zone
    for zone_name, file_name in REGION_BATCHES:
        locs = [l for l in all_locs if l["agro_zone"] == zone_name]
        if not locs:
            print(f"WARNING: No locations for zone '{zone_name}'")
            continue
        generate_batch(locs, file_name, rows_per_season=ROWS_PER_SEASON_BASE)

    # Rest of India (non-Maharashtra)
    roi = [l for l in all_locs if l["state"] != "Maharashtra"]
    generate_batch(roi, "Rest_of_India", rows_per_season=ROWS_PER_SEASON_BASE)

    # Merge all + validate
    print("\n" + "=" * 60)
    print("MERGING ALL BATCHES")
    print("=" * 60)
    merge_and_validate()


if __name__ == "__main__":
    main()


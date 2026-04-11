"""
Streamlit Dashboard for the Crop Recommendation Engine.

Usage:
    1. Start the API server:
       cd Crop_Recommendation_Engine && python -c "import uvicorn; uvicorn.run('api:app', host='127.0.0.1', port=8002)"
    2. Run the dashboard:
       streamlit run Crop_Recommendation_Engine/app.py --server.port 8501
"""

import json
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

# Use gateway for all requests (works both locally and via tunnel)
API_BASE = "http://127.0.0.1:8080/api/crop"

# ─── Constants ───
SOIL_TYPES = [
    "Black (Regur)", "Red", "Laterite", "Alluvial",
    "Sandy", "Clay", "Medium Black", "Shallow Black",
]
SOIL_TEXTURES = ["Clay", "Clay Loam", "Silty Clay", "Sandy Loam", "Loam", "Sandy Clay"]
DRAINAGE_CLASSES = ["Poor", "Moderate", "Good"]
AGRO_ZONES = ["Vidarbha", "Marathwada", "Western Maharashtra", "Konkan", "Northern Maharashtra"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
IRRIGATION_TYPES = ["Rainfed", "Drip", "Sprinkler", "Flood"]
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Preset locations in Maharashtra
PRESET_LOCATIONS = {
    "📍 Click on map": (19.5, 76.5),
    "Nagpur (Vidarbha)": (21.15, 79.09),
    "Amravati (Vidarbha)": (20.93, 77.75),
    "Aurangabad (Marathwada)": (19.88, 75.34),
    "Latur (Marathwada)": (18.40, 76.57),
    "Pune (Western MH)": (18.52, 73.86),
    "Kolhapur (Western MH)": (16.70, 74.24),
    "Nashik (Northern MH)": (20.00, 73.78),
    "Ratnagiri (Konkan)": (16.99, 73.30),
}

# ─── Agro Zone Bounding Boxes (for auto-detection from coordinates) ───
AGRO_ZONE_BOUNDS = {
    "Vidarbha":              {"lat": (19.5, 22.0), "lon": (76.0, 80.5)},
    "Marathwada":            {"lat": (17.5, 20.5), "lon": (74.5, 77.5)},
    "Western Maharashtra":   {"lat": (15.5, 19.5), "lon": (73.5, 76.0)},
    "Konkan":                {"lat": (15.5, 20.0), "lon": (72.5, 74.0)},
    "Northern Maharashtra":  {"lat": (19.5, 22.0), "lon": (72.5, 76.0)},
}


def _detect_agro_zone(lat: float, lon: float) -> str:
    """Auto-detect agro zone from coordinates using bounding boxes.

    Returns the best-matching zone name, or empty string if outside all zones.
    When coordinates fall in overlapping zones, returns the smallest (most specific) zone.
    """
    matches = []
    for zone, bounds in AGRO_ZONE_BOUNDS.items():
        lat_lo, lat_hi = bounds["lat"]
        lon_lo, lon_hi = bounds["lon"]
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            area = (lat_hi - lat_lo) * (lon_hi - lon_lo)
            matches.append((zone, area))
    if not matches:
        return ""
    # Return smallest bounding box (most specific match)
    matches.sort(key=lambda x: x[1])
    return matches[0][0]

# Map agro zone from location
LOCATION_ZONE = {
    "Nagpur (Vidarbha)": "Vidarbha",
    "Amravati (Vidarbha)": "Vidarbha",
    "Aurangabad (Marathwada)": "Marathwada",
    "Latur (Marathwada)": "Marathwada",
    "Pune (Western MH)": "Western Maharashtra",
    "Kolhapur (Western MH)": "Western Maharashtra",
    "Nashik (Northern MH)": "Northern Maharashtra",
    "Ratnagiri (Konkan)": "Konkan",
}


# ─── Page Config ───
st.set_page_config(
    page_title="🌾 Crop Recommender — Maharashtra",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for comprehensive UI fixes
st.markdown("""
<style>
    /* Compact and left-aligned sidebar */
    .stSidebar {
        width: 250px !important;
        min-width: 250px !important;
    }
    
    .stSidebar [data-testid="stSidebarContent"] {
        width: 250px !important;
        padding: 1rem 1rem !important;
    }
    
    /* Compact sidebar spacing */
    .stSidebar > div {
        gap: 0.5rem !important;
    }
    
    .stSidebar [data-testid="stSidebarContent"] > [data-testid="stColumnContainer"] {
        gap: 0.5rem !important;
    }
    
    /* Left-align all sidebar content */
    .stSidebar * {
        color: inherit !important;
        text-align: left !important;
    }
    
    .stSidebar h3, .stSidebar h2 {
        color: currentColor !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1rem !important;
    }
    
    .stSidebar h1 {
        font-size: 0.9rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    .stSidebar label {
        color: currentColor !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.2rem !important;
    }
    
    .stSidebar p {
        font-size: 0.85rem !important;
        margin: 0.3rem 0 !important;
    }
    
    /* Compact input fields in sidebar */
    .stSidebar input, .stSidebar select {
        padding: 6px 8px !important;
        font-size: 0.85rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Reduce button size in sidebar */
    .stSidebar button {
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
        margin-top: 0.3rem !important;
    }
    
    /* Main content padding fix */
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Map container - proper sizing in box */
    div[data-testid="stIFrame"] {
        width: 100% !important;
        height: 450px !important;
        border: 2px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        margin: 1rem 0 !important;
    }
    
    /* Folium map styling */
    .folium-map {
        width: 100% !important;
        height: 450px !important;
        border-radius: 10px !important;
    }
    
    /* Input fields visibility and styling */
    input {
        padding: 10px 12px !important;
        border-radius: 6px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        font-size: 0.95rem !important;
    }
    
    input:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
    }
    
    select {
        padding: 10px 12px !important;
        border-radius: 6px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Metrics styling - light & dark compatible */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.5) !important;
        border-radius: 10px !important;
        padding: 15px 18px !important;
        border-left: 4px solid #4CAF50 !important;
        backdrop-filter: blur(4px) !important;
    }
    
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        opacity: 0.75 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 2px solid rgba(76, 175, 80, 0.2) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        border-radius: 6px 6px 0 0 !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4CAF50 !important;
        color: white !important;
        border-color: #4CAF50 !important;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.8rem !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton > button {
        padding: 10px 24px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        border: none !important;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
        color: white !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(76, 175, 80, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0 !important;
        opacity: 0.2 !important;
    }
    
    /* Container styling */
    .stContainer {
        border-radius: 8px !important;
    }
    
    /* Success/Error/Info messages */
    .stAlert {
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Sidebar expandable items */
    .stExpander {
        border-radius: 8px !important;
    }
    
    /* Ensure all text is readable */
    body, p, span, a, li {
        color: inherit !important;
    }
    
    /* Form label styling */
    .stForm label {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── API helper ───


# ─── API helper ───
def _api_call(method, endpoint, payload=None, params=None):
    """Make API call and return (success, data_or_error)."""
    try:
        if method == "GET":
            r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=20)
        else:
            r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=20)
        if r.status_code == 200:
            return True, r.json()
        return False, f"HTTP {r.status_code}: {r.text[:500]}"
    except Exception as e:
        return False, str(e)


# ─── Header ───
st.title("🌾 Crop Recommendation Engine")
st.caption("AI-powered crop advisory for Maharashtra, India — powered by a calibrated Random Forest model")

# ── Check API health ──
health = None
try:
    r = requests.get(f"{API_BASE}/health", timeout=5)
    if r.status_code == 200:
        health = r.json()
    else:
        error_detail = r.json().get('detail', r.text) if r.headers.get('content-type') == 'application/json' else r.text
        st.error(f"⚠️ **API returned error ({r.status_code}):** {error_detail}")
        st.info("**Fix:** Ensure API Gateway is running:\n```\ncd ApiGateway\npython main.py\n```")
        st.stop()
except Exception as e:
    st.error("⚠️ **API server not reachable.** Please ensure:")
    st.code("Terminal 1: keep_tunnel_gateway_alive.bat\nTerminal 2: AgroManager.bat → [1] Start ALL")
    st.stop()

if not health:
    st.error("❌ Failed to fetch API health data")
    st.stop()


# ╔══════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — Inputs                                           ║
# ╚══════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.image("https://img.icons8.com/color/96/wheat.png", width=60)
    st.markdown("### Input Parameters")
    num_crops = health.get('num_classes', '?')
    st.success(f"✅ API Online • {num_crops} crops")

    # ── 1. LOCATION ──
    st.markdown("---")
    st.markdown("#### 📍 Location")

    # Initialize session state for map-clicked coordinates
    if "map_lat" not in st.session_state:
        st.session_state.map_lat = None
    if "map_lon" not in st.session_state:
        st.session_state.map_lon = None
    if "map_zone" not in st.session_state:
        st.session_state.map_zone = None

    _loc_keys = list(PRESET_LOCATIONS.keys())
    # Auto-switch to "Click on map" when user applied map coordinates
    if st.session_state.map_lat is not None:
        _default_idx = 0  # "📍 Click on map" is first entry
    else:
        _default_idx = _loc_keys.index("Kolhapur (Western MH)") if "Kolhapur (Western MH)" in _loc_keys else 0
    preset = st.selectbox(
        "Quick Location",
        _loc_keys,
        index=_default_idx,
        help="Select a known city, or click on the map and press 'Use this location'",
    )

    # Determine lat/lon: map click > preset
    if preset == "📍 Click on map" and st.session_state.map_lat is not None:
        default_lat = st.session_state.map_lat
        default_lon = st.session_state.map_lon
    else:
        default_lat, default_lon = PRESET_LOCATIONS[preset]
        # Clear map state when user switches to a preset
        if preset != "📍 Click on map":
            st.session_state.map_lat = None
            st.session_state.map_lon = None
            st.session_state.map_zone = None

    # Auto-set agro zone: map click > preset lookup > manual
    if preset == "📍 Click on map" and st.session_state.map_zone:
        auto_zone = st.session_state.map_zone
    else:
        auto_zone = LOCATION_ZONE.get(preset, None)

    lat = st.number_input("Latitude", 15.5, 22.5, default_lat, 0.01, format="%.2f")
    lon = st.number_input("Longitude", 72.5, 80.5, default_lon, 0.01, format="%.2f")
    altitude = st.number_input("Altitude (m)", 0, 2000, 350, 10)

    if auto_zone:
        zone_idx = AGRO_ZONES.index(auto_zone) if auto_zone in AGRO_ZONES else 0
        agro_zone = st.selectbox("Agro Zone", AGRO_ZONES, index=zone_idx)
    else:
        agro_zone = st.selectbox("Agro Zone", AGRO_ZONES)

    # ── 2. SEASON ──
    st.markdown("---")
    st.markdown("#### 📅 Season & Timing")
    season = st.selectbox("Season", SEASONS)
    month = st.select_slider("Month", options=list(range(1, 13)),
                              format_func=lambda x: MONTH_NAMES[x], value=7)

    # ── 3. SOIL ──
    st.markdown("---")
    st.markdown("#### 🧪 Soil Properties")

    col1, col2 = st.columns(2)
    nitrogen = col1.number_input("Nitrogen (N)", 0, 500, 110, help="mg/kg")
    phosphorus = col2.number_input("Phosphorus (P)", 0, 300, 55, help="mg/kg")

    col3, col4 = st.columns(2)
    potassium = col3.number_input("Potassium (K)", 0, 500, 70, help="mg/kg")
    temperature = col4.number_input("Soil Temp °C", 0.0, 55.0, 26.0, 0.5)

    col5, col6 = st.columns(2)
    moisture = col5.number_input("Moisture %", 0.0, 100.0, 55.0, 1.0)
    ec = col6.number_input("EC (μS/cm)", 0, 20000, 1400, 50)

    ph = st.slider("pH", 3.0, 10.0, 7.5, 0.1)
    organic_carbon = st.slider("Organic Carbon %", 0.0, 5.0, 0.8, 0.1)

    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    soil_texture = st.selectbox("Soil Texture", SOIL_TEXTURES, index=1)
    drainage = st.selectbox("Drainage", DRAINAGE_CLASSES, index=1)

    # ── 4. WEATHER (manual) ──
    st.markdown("---")
    st.markdown("#### 🌤️ Weather")
    use_live = st.checkbox("Auto-fetch weather from Open-Meteo", value=True)

    if not use_live:
        weather_temp = st.number_input("Air Temp °C", -5.0, 55.0, 28.0, 0.5)
        humidity = st.number_input("Humidity %", 0.0, 100.0, 72.0, 1.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 900.0, 10.0)
        sunshine = st.number_input("Sunshine hrs/day", 0.0, 14.0, 5.5, 0.5)
        wind_speed = st.number_input("Wind km/h", 0.0, 100.0, 8.0, 0.5)
    else:
        weather_temp, humidity, rainfall, sunshine, wind_speed = 28.0, 72.0, 900.0, 5.5, 8.0

    # ── 5. OPTIONAL ──
    st.markdown("---")
    st.markdown("#### 🔄 Optional")
    prev_crop = st.text_input("Previous Crop", placeholder="e.g. Soybean, Cotton")
    irrigation_type = st.selectbox("Irrigation Type", ["None"] + IRRIGATION_TYPES)


# ─── Build payload ───
base_payload = dict(
    nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
    temperature=temperature, moisture=moisture, ec=ec, ph=ph,
    lat=lat, lon=lon, altitude=altitude, organic_carbon=organic_carbon,
    soil_type=soil_type, soil_texture=soil_texture, drainage=drainage,
    agro_zone=agro_zone, season=season, month=month,
)
if prev_crop.strip():
    base_payload["prev_crop"] = prev_crop.strip()
if irrigation_type != "None":
    base_payload["irrigation_type"] = irrigation_type


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN AREA — Map + Tabs                                     ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Interactive Clickable Map ──
with st.expander("🗺️ **Click Map to Select Location**", expanded=True):
    st.caption("👆 **Click anywhere on the map**, then press the button below to apply coordinates.")

    # Build folium map — NO popups on overlays so clicks pass through
    m = folium.Map(
        location=[float(lat), float(lon)],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    # Agro zone overlays — click-through (no popup, no tooltip)
    _zone_colors = {
        "Vidarbha": "#E91E63", "Marathwada": "#FF9800",
        "Western Maharashtra": "#2196F3", "Konkan": "#009688",
        "Northern Maharashtra": "#9C27B0",
    }
    for zone_name, bounds in AGRO_ZONE_BOUNDS.items():
        lat_lo, lat_hi = bounds["lat"]
        lon_lo, lon_hi = bounds["lon"]
        rect = [[lat_lo, lon_lo], [lat_lo, lon_hi], [lat_hi, lon_hi], [lat_hi, lon_lo]]
        color = _zone_colors.get(zone_name, "#666666")
        folium.Rectangle(
            bounds=[[lat_lo, lon_lo], [lat_hi, lon_hi]],
            color=color, weight=1, fill=True, fill_opacity=0.06,
        ).add_to(m)
        # Label in center
        folium.Marker(
            location=[(lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2],
            icon=folium.DivIcon(html=f'<div style="font-size:10px;color:{color};font-weight:bold;'
                                     f'white-space:nowrap">{zone_name}</div>'),
        ).add_to(m)

    # Current location marker
    folium.Marker(
        [float(lat), float(lon)],
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Render map and capture clicks
    map_data = st_folium(m, width=700, height=400)

    # Show click result and "Apply" button
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        clicked_lat = round(clicked["lat"], 2)
        clicked_lon = round(clicked["lng"], 2)
        detected_zone = _detect_agro_zone(clicked_lat, clicked_lon)
        zone_label = detected_zone if detected_zone else "Unknown zone"

        st.markdown(f"📌 **Clicked:** `{clicked_lat}°N, {clicked_lon}°E` → **{zone_label}**")

        if st.button("📍 Use this location", type="primary", use_container_width=True):
            st.session_state.map_lat = clicked_lat
            st.session_state.map_lon = clicked_lon
            st.session_state.map_zone = detected_zone if detected_zone else None
            st.rerun()
    else:
        st.info("💡 Click on the map to pick a location. A pin + apply button will appear.")

st.markdown("---")

# ── Prediction display helpers ──
def _show_prediction(result: dict):
    """Render prediction results in a user-friendly layout."""
    flag = result.get("confidence_flag", "?")
    flag_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠"}.get(flag, "🔴")
    st.markdown(f"{flag_icon} **Overall Confidence: {flag}**")

    # Warnings
    if result.get("is_ood"):
        st.warning("⚠️ Out-of-distribution input — predictions may be unreliable.")
    for w in result.get("input_warnings", []):
        st.warning(w)

    # Advisory
    st.info(f"📝 {result.get('advisory', 'No advisory available.')}")

    # Top 3 crops
    top3 = result.get("top_3", [])
    if top3:
        cols = st.columns(len(top3))
        for i, crop in enumerate(top3):
            with cols[i]:
                conf = crop.get("confidence", 0)
                pct = crop.get("confidence_pct", "?")
                crop_flag = crop.get("flag", "?")
                flag_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠"}.get(crop_flag, "🔴")

                st.markdown(f"### {flag_icon} #{i+1}")
                st.markdown(f"## {crop['crop']}")
                st.metric("Confidence", pct)
                st.caption(f"Season: {crop.get('season', '?')} | {crop_flag}")
                if crop.get("rotation_note"):
                    st.caption(f"🔄 {crop['rotation_note']}")

                # Guardrail notes (surfaced)
                g_notes = crop.get("guardrail_notes", [])
                if g_notes:
                    with st.expander(f"🛡️ Guardrail notes ({len(g_notes)})"):
                        for note in g_notes:
                            if "penalty" in note:
                                st.markdown(f"🔻 {note}")
                            elif "boost" in note:
                                st.markdown(f"🔺 {note}")
                            else:
                                st.markdown(f"ℹ️ {note}")

                # Progress bar
                st.progress(min(conf, 1.0))

    # ── Farmer's Advisory Panel ──
    advisory = result.get("farmer_advisory")
    if advisory:
        _show_farmer_advisory(advisory)


def _show_farmer_advisory(advisory: dict):
    """Render the structured farmer advisory panel."""
    st.markdown("---")
    st.markdown("### 🧑‍🌾 Farmer's Advisory")

    # Why this crop
    why = advisory.get("why_this_crop", "")
    if why:
        st.success(f"🌱 **Why this crop?** {why}")

    # Warnings
    warnings = advisory.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w)

    # Two-column layout: Irrigation + Next Crop
    col_left, col_right = st.columns(2)
    with col_left:
        irr = advisory.get("irrigation_tips", "")
        if irr:
            st.info(f"💧 **Irrigation Guidance**\n\n{irr}")

    with col_right:
        next_crop = advisory.get("next_crop", "")
        if next_crop:
            st.info(f"🔄 **What to Plant Next**\n\n{next_crop}")

    # Sowing window
    sowing = advisory.get("sowing_window", "")
    if sowing:
        st.caption(f"📅 {sowing}")

    # Soil health assessment
    soil_health = advisory.get("soil_health", {})
    if soil_health:
        with st.expander("🧪 Soil Health Assessment"):
            sh_cols = st.columns(5)
            param_icons = {"nitrogen": "🟢", "phosphorus": "🟠", "potassium": "🔵", "ph": "⚗️", "ec": "⚡"}
            for i, (param, info) in enumerate(soil_health.items()):
                with sh_cols[i]:
                    status = info.get("status", "?")
                    val = info.get("value", "?")
                    ideal = info.get("ideal_range", "?")
                    icon = param_icons.get(param, "📊")
                    status_color = {"Low": "🔴", "Adequate": "🟢", "High": "🟡", "OK": "🟢", "Concern": "🔴"}.get(status, "⚪")
                    st.markdown(f"**{icon} {param.upper()}**")
                    st.metric(f"{status_color} {status}", f"{val}")
                    if ideal != "?":
                        st.caption(f"Ideal: {ideal}")


def _show_full_json(input_params: dict, api_response: dict, label: str = "Crop Recommendation"):
    """Show combined input+output JSON and a ready-to-paste LLM prompt."""
    combined = {
        "input_parameters": input_params,
        "model_output": api_response,
    }
    with st.expander("🔍 View Full Data (Input + Output)"):
        st.code(json.dumps(combined, indent=2), language="json")

    # Build LLM prompt
    prompt_text = (
        f"I'm using a crop recommendation model for Maharashtra, India. "
        f"Please cross-validate the following {label} result.\n\n"
        f"## Input Parameters (Sensor + Location + Weather)\n"
        f"```json\n{json.dumps(input_params, indent=2)}\n```\n\n"
        f"## Model Output\n"
        f"```json\n{json.dumps(api_response, indent=2)}\n```\n\n"
        f"Based on the above soil, weather, and location data, do you agree with the model's "
        f"recommendation? What crop would you suggest and why? Please highlight any concerns."
    )
    with st.expander("📋 Copy for LLM Cross-Validation"):
        st.caption("Click the 📋 icon in the top-right corner to copy:")
        st.code(prompt_text, language="markdown")


# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Get Recommendation",
    "🌍 Live Weather Prediction",
    "🔄 Yearly Rotation Plan",
    "🧪 Fertilizer Calculator",
    "☁️ Current Weather",
])

# ── Tab 1: Prediction ──
with tab1:
    st.markdown("### 🎯 Crop Recommendation")
    st.caption("Uses your soil parameters, location, and weather data to recommend the best crops.")

    if use_live:
        st.info("💡 Weather will be auto-fetched from Open-Meteo. Switch off in sidebar to enter manually.")

    if st.button("🚀 Get Recommendation", key="predict_btn", type="primary", use_container_width=True):
        with st.spinner("Analyzing soil and weather data..."):
            if use_live:
                ok, data = _api_call("POST", "/predict/live", base_payload)
            else:
                payload = {**base_payload, "weather_temp": weather_temp, "humidity": humidity,
                           "rainfall": rainfall, "sunshine": sunshine, "wind_speed": wind_speed}
                ok, data = _api_call("POST", "/predict", payload)

            if ok:
                _show_prediction(data)
                sent_payload = payload if not use_live else base_payload
                _show_full_json(sent_payload, data, "Crop Recommendation")
            else:
                st.error(f"❌ Prediction failed: {data}")

# ── Tab 2: Live Prediction ──
with tab2:
    st.markdown("### 🌍 Prediction with Live Weather")
    st.caption("Automatically fetches current weather from Open-Meteo API for your location.")
    st.markdown(f"**Location:** {lat:.2f}°N, {lon:.2f}°E ({agro_zone})")

    if st.button("🌍 Predict with Live Weather", key="live_btn", type="primary", use_container_width=True):
        with st.spinner("Fetching weather and analyzing..."):
            ok, data = _api_call("POST", "/predict/live", base_payload)
            if ok:
                # Show fetched weather if available
                if data.get("weather_used"):
                    wu = data["weather_used"]
                    wcols = st.columns(5)
                    wcols[0].metric("🌡️ Temp", f"{wu.get('weather_temp', '?')}°C")
                    wcols[1].metric("💧 Humidity", f"{wu.get('humidity', '?')}%")
                    wcols[2].metric("🌧️ Rain", f"{wu.get('rainfall', '?')} mm")
                    wcols[3].metric("☀️ Sun", f"{wu.get('sunshine', '?')} hrs")
                    wcols[4].metric("💨 Wind", f"{wu.get('wind_speed', '?')} km/h")
                    st.markdown("---")
                _show_prediction(data)
                _show_full_json(base_payload, data, "Live Weather Prediction")
            else:
                st.error(f"❌ {data}")

# ── Tab 3: Rotation ──
with tab3:
    st.markdown("### 🔄 Full-Year Crop Rotation Plan")
    st.caption("Generates an optimized Kharif → Rabi → Zaid rotation based on your soil and location.")

    if st.button("🔄 Generate Rotation Plan", key="rotation_btn", type="primary", use_container_width=True):
        with st.spinner("Planning optimal rotation..."):
            rot_payload = {k: v for k, v in base_payload.items()
                           if k not in ("season", "month", "prev_crop", "irrigation_type")}
            ok, data = _api_call("POST", "/rotation", rot_payload)
            if ok:
                season_icons = {"Kharif": "🌧️", "Rabi": "❄️", "Zaid": "☀️"}
                rotation = data.get("rotation", [])
                cols = st.columns(len(rotation))
                for i, entry in enumerate(rotation):
                    with cols[i]:
                        s = entry["season"]
                        rec = entry["recommendation"]
                        top = rec["top_3"][0] if rec["top_3"] else {}
                        chosen = entry.get("chosen_crop", top.get("crop", "?"))
                        conf_pct = top.get("confidence_pct", "?")
                        flag = top.get("flag", "?")

                        st.markdown(f"### {season_icons.get(s, '📅')} {s}")
                        st.markdown(f"## {chosen}")
                        st.metric("Confidence", conf_pct)
                        st.caption(rec.get("advisory", "")[:120])

                _show_full_json(rot_payload, data, "Rotation Plan")
            else:
                st.error(f"❌ {data}")

# ── Tab 4: Fertilizer Calculator ──
with tab4:
    st.markdown("### 🧪 Fertilizer Amendment Calculator")
    st.caption("Calculate NPK gaps and get specific fertilizer recommendations for your target crop.")

    ok_crops, crop_data = _api_call("GET", "/crops")
    crop_names = [c["name"] for c in crop_data.get("crops", [])] if ok_crops else ["Soybean"]

    col_a, col_b = st.columns(2)
    target_crop = col_a.selectbox("Target Crop", crop_names, key="amend_crop")
    field_area = col_b.number_input("Field Area (hectares)", 0.1, 100.0, 1.0, 0.1)

    st.markdown(f"**Current Soil NPK:** N={nitrogen}, P={phosphorus}, K={potassium} mg/kg")

    if st.button("🧪 Calculate Amendments", key="amend_btn", type="primary", use_container_width=True):
        with st.spinner("Calculating nutrient gaps..."):
            amend_payload = dict(
                crop_name=target_crop,
                nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium,
                field_area_ha=field_area,
            )
            ok, data = _api_call("POST", "/amendments", amend_payload)
            if ok:
                st.markdown(f"### 🌱 Amendment Plan for **{data.get('crop', target_crop)}**")
                st.divider()

                # ── NPK Comparison: Current vs Ideal ──
                st.markdown("#### 📊 Nutrient Comparison (mg/kg)")
                current = data.get("current_npk", {})
                ideal = data.get("ideal_npk", {})
                gap = data.get("gap_npk", {})

                npk_cols = st.columns(3)
                for i, nutrient in enumerate(["N", "P", "K"]):
                    with npk_cols[i]:
                        cur_val = current.get(nutrient, 0)
                        ideal_val = ideal.get(nutrient, 0)
                        gap_val = gap.get(nutrient, 0)
                        if gap_val > 0:
                            delta_str = f"⚠️ Deficit: {gap_val:.0f}"
                            d_color = "inverse"
                        else:
                            delta_str = "✅ Sufficient"
                            d_color = "normal"
                        nutrient_name = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}[nutrient]
                        st.metric(
                            f"{nutrient_name} ({nutrient})",
                            f"{cur_val:.0f} / {ideal_val:.0f}",
                            delta=delta_str,
                            delta_color=d_color,
                        )

                st.divider()

                # ── Fertilizer Recommendations ──
                fert = data.get("fertilizer_kg_per_ha", {})
                total = data.get("total_for_field", {})
                area = total.get("field_area_ha", field_area)

                if any(v > 0 for v in fert.values()):
                    st.markdown("#### 🧪 Recommended Fertilizers")
                    fert_cols = st.columns(3)
                    fert_info = {
                        "Urea": {"icon": "🟡", "nutrient": "46% N", "per_ha": fert.get("Urea", 0),
                                 "total": total.get("Urea_kg", 0)},
                        "DAP":  {"icon": "🟤", "nutrient": "20% P", "per_ha": fert.get("DAP", 0),
                                 "total": total.get("DAP_kg", 0)},
                        "MOP":  {"icon": "🔴", "nutrient": "50% K", "per_ha": fert.get("MOP", 0),
                                 "total": total.get("MOP_kg", 0)},
                    }
                    for i, (name, info) in enumerate(fert_info.items()):
                        with fert_cols[i]:
                            st.markdown(f"**{info['icon']} {name}**")
                            st.markdown(f"*({info['nutrient']})*")
                            if info["per_ha"] > 0:
                                st.metric("Per Hectare", f"{info['per_ha']:.1f} kg")
                                if area != 1.0:
                                    st.metric(f"Total ({area:.1f} ha)", f"{info['total']:.1f} kg")
                            else:
                                st.markdown("✅ *Not needed*")
                else:
                    st.success("🎉 **No amendments needed!** Your soil NPK already meets or exceeds the crop's requirements.")

                # ── Notes ──
                notes = data.get("notes", [])
                if notes:
                    st.divider()
                    st.markdown("#### 📝 Notes")
                    for note in notes:
                        st.info(f"💡 {note}")

                _show_full_json(amend_payload, data, "Fertilizer Amendment")
            else:
                st.error(f"❌ {data}")

# ── Tab 5: Weather ──
with tab5:
    st.markdown("### ☁️ Current Weather")
    st.caption("Fetch real-time weather data from Open-Meteo for any location.")

    w_col1, w_col2 = st.columns(2)
    w_lat = w_col1.number_input("Latitude", 5.0, 40.0, float(lat), 0.1, key="w_lat")
    w_lon = w_col2.number_input("Longitude", 65.0, 100.0, float(lon), 0.1, key="w_lon")

    if st.button("☁️ Fetch Weather", key="weather_btn", type="primary", use_container_width=True):
        with st.spinner("Fetching weather data..."):
            ok, data = _api_call("GET", "/weather", params={"lat": w_lat, "lon": w_lon})
            if ok:
                w = data.get("weather", {})
                wcols = st.columns(5)
                wcols[0].metric("🌡️ Temperature", f"{w.get('weather_temp', '?')}°C")
                wcols[1].metric("💧 Humidity", f"{w.get('humidity', '?')}%")
                wcols[2].metric("🌧️ Rainfall", f"{w.get('rainfall', '?')} mm")
                wcols[3].metric("☀️ Sunshine", f"{w.get('sunshine', '?')} hrs")
                wcols[4].metric("💨 Wind Speed", f"{w.get('wind_speed', '?')} km/h")

                _show_full_json({"lat": w_lat, "lon": w_lon}, data, "Weather Fetch")
            else:
                st.error(f"❌ {data}")

# ── Footer ──
st.markdown("---")
num_crops = health.get('num_classes', '?') if health else '?'
st.caption("🌾 Crop Recommendation Engine v2.0 • Maharashtra, India • "
           f"Model: Calibrated Random Forest ({num_crops} crops)")

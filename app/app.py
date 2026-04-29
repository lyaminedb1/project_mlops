"""
Waste Detection — Streamlit UI

Features:
- Model selector + image uploader + GPS inputs → POST /predict
- Folium map of all detections from GET /history
- Sidebar filters: source, model, date range
"""

import os
from datetime import date, datetime, timedelta

import folium
import requests
import streamlit as st
from streamlit_folium import folium_static

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Waste Detection", layout="wide")
st.title("Urban Waste Detection — Drone Patrol")


# ── Helpers ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json()]
    except Exception:
        return []


@st.cache_data(ttl=10)
def fetch_history():
    try:
        resp = requests.get(f"{API_URL}/history", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# ── Sidebar filters ──────────────────────────────────────────────────────────────

st.sidebar.header("Filters")

source_filter = st.sidebar.selectbox(
    "Source",
    options=["all", "manual", "drone_patrol"],
    index=0,
)

all_models = fetch_models()
model_filter = st.sidebar.multiselect(
    "Model",
    options=all_models,
    default=[],
    placeholder="All models",
)

default_start = date.today() - timedelta(days=30)
date_start = st.sidebar.date_input("From", value=default_start)
date_end = st.sidebar.date_input("To", value=date.today())


# ── Layout: two columns ──────────────────────────────────────────────────────────

col_form, col_map = st.columns([1, 2])

# ── Left column: prediction form ─────────────────────────────────────────────────

with col_form:
    st.subheader("Submit a detection")

    selected_model = st.selectbox(
        "Model",
        options=all_models if all_models else ["(no models available)"],
    )
    uploaded_file = st.file_uploader("Image (JPEG / PNG, max 10 MB)", type=["jpg", "jpeg", "png"])
    latitude = st.text_input("Latitude", value="48.8566")
    longitude = st.text_input("Longitude", value="2.3522")

    if st.button("Run detection", type="primary"):
        if not uploaded_file:
            st.warning("Please upload an image first.")
        elif not all_models:
            st.error("No models available — is the API running?")
        else:
            with st.spinner("Running inference…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        data={
                            "latitude": latitude,
                            "longitude": longitude,
                            "model_name": selected_model,
                        },
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success("Detection complete")
                        st.metric("Confidence", f"{result['confiance']:.2%}")
                        st.json(result)
                        st.cache_data.clear()
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as exc:
                    st.error(f"Could not reach API: {exc}")


# ── Right column: map ─────────────────────────────────────────────────────────────

with col_map:
    st.subheader("Detection map")

    detections = fetch_history()

    # Apply filters
    def _passes_filter(d: dict) -> bool:
        if source_filter != "all" and d.get("source") != source_filter:
            return False
        if model_filter and d.get("model_name") not in model_filter:
            return False
        ts = d.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            if dt < date_start or dt > date_end:
                return False
        except Exception:
            pass
        return True

    filtered = [d for d in detections if _passes_filter(d)]

    m = folium.Map(location=[48.8566, 2.3522], zoom_start=5)

    for d in filtered:
        lat = d.get("latitude")
        lon = d.get("longitude")
        if lat is None or lon is None:
            continue

        color = "red" if d.get("source") == "manual" else "orange"
        popup_html = (
            f"<b>Time:</b> {d.get('timestamp', '')}<br>"
            f"<b>Confidence:</b> {d.get('confiance', 0):.2%}<br>"
            f"<b>Model:</b> {d.get('model_name', '')}<br>"
            f"<b>Source:</b> {d.get('source', '')}"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon="trash", prefix="fa"),
        ).add_to(m)

    folium_static(m, width=700, height=500)

    st.caption(
        f"Showing **{len(filtered)}** of **{len(detections)}** detections. "
        "Red = manual, Orange = drone patrol."
    )

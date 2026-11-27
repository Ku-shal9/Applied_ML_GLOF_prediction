## streamlit dashboard for glof risk prediction and animated outburst simulation

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import FEATURES, METADATA_PATH, MODEL_PATH, THRESHOLD
from src.predict import load_model, predict


# load model and metadata once and cache across interactions
@st.cache_resource
def get_model_and_meta():
    model = load_model(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


# build animated plotly cross-section of moraine dam and downstream flood wave
def build_glof_animation(mean_depth, volume_dam, volume_gap, expansion_frac, proba):
    x = np.linspace(0, 10, 160)
    # terrain: lake basin (0..3.5), moraine dam (3.5..4.5), downstream valley (4.5..10)
    dam_height = 22.0
    terrain = np.where(
        x < 3.5,
        2.0,
        np.where(
            x <= 4.5,
            dam_height * np.exp(-((x - 4.0) ** 2) / 0.12),
            np.maximum(0.5, 4.0 - 0.35 * (x - 4.5)),
        ),
    )

    is_critical = proba >= THRESHOLD
    # scale initial lake level by mean_depth and expansion_frac
    base_level = np.clip(10.0 + 0.35 * mean_depth + max(expansion_frac, 0.0) * 6.0, 10.0, 21.2)
    surge_scale = np.clip((np.log1p(volume_dam) / 5.0) * (0.5 + proba), 0.4, 2.2)

    frames = []
    steps = 24
    for step in range(steps):
        t_frac = step / (steps - 1)
        breach_t = max(0.0, (t_frac - 0.15) / 0.85) if is_critical else 0.0

        # erode moraine crest when critical
        eroded_dam = terrain.copy()
        if is_critical:
            crest_mask = (x >= 3.5) & (x <= 4.5)
            eroded_dam[crest_mask] -= 9.5 * breach_t * np.exp(-((x[crest_mask] - 4.0) ** 2) / 0.1)

        # lake water drops as flood drains downstream
        lake_level = base_level - (6.5 * breach_t if is_critical else 0.0)
        water = np.full_like(x, np.nan)
        lake_mask = x <= 4.0
        water[lake_mask] = np.maximum(eroded_dam[lake_mask], lake_level)

        if is_critical and breach_t > 0:
            wave_front = 4.0 + 6.2 * breach_t
            downstream_mask = (x > 4.0) & (x <= wave_front)
            wave_profile = (
                eroded_dam[downstream_mask]
                + 5.5 * surge_scale * np.exp(-((x[downstream_mask] - wave_front + 0.8) ** 2) / 1.4)
                + 1.2 * surge_scale * (1.0 - breach_t * 0.4)
            )
            water[downstream_mask] = wave_profile
        else:
            downstream_mask = x > 4.2
            water[downstream_mask] = eroded_dam[downstream_mask] + 0.4

        # village marker height at x = 8.5
        village_y = float(np.interp(8.5, x, eroded_dam))
        flood_at_village = float(np.interp(8.5, x, np.nan_to_num(water, nan=0.0))) - village_y
        village_color = "#e11d48" if flood_at_village > 0.8 else "#16a34a"

        frames.append(
            go.Frame(
                name=f"{int(t_frac * 60)} min",
                data=[
                    go.Scatter(x=x, y=eroded_dam, fill="tozeroy", fillcolor="rgba(120, 113, 108, 0.75)", line=dict(color="#57534e", width=2), name="Moraine & Valley"),
                    go.Scatter(x=x, y=water, fill="tonexty", fillcolor="rgba(14, 165, 233, 0.55)" if not is_critical else "rgba(239, 68, 68, 0.55)", line=dict(color="#0284c7" if not is_critical else "#dc2626", width=2), name="Water / Surge"),
                    go.Scatter(x=[8.5], y=[village_y + 0.8], mode="markers+text", text=["Downstream Settlement"], textposition="top center", marker=dict(size=14, color=village_color, symbol="triangle-up"), name="Settlement"),
                ],
            )
        )

    init_data = frames[0].data
    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        title="GLOF Moraine Breach & Downstream Surge Simulation (0 to 60 min)",
        xaxis_title="Distance Along Valley (km)",
        yaxis_title="Elevation / Water Depth (m)",
        yaxis=dict(range=[0, 26]),
        xaxis=dict(range=[0, 10]),
        height=480,
        margin=dict(l=20, r=20, t=60, b=115),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.0,
                "xanchor": "left",
                "y": -0.22,
                "yanchor": "top",
                "pad": {"r": 10, "t": 10},
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ],
        sliders=[
            {
                "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": f.name, "method": "animate"} for f in frames],
                "x": 0.22,
                "len": 0.78,
                "y": -0.20,
                "yanchor": "top",
                "pad": {"b": 10, "t": 10},
            }
        ],
    )
    return fig


# render streamlit dashboard
def main():
    st.set_page_config(page_title="GLOF Risk Dashboard", layout="wide")
    st.title("Glacial Lake Outburst Flood (GLOF) Risk & Simulation Dashboard")

    model, meta = get_model_and_meta()
    metrics = meta["metrics"]

    # top model summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AP", f"{metrics['average_precision']:.3f}")
    c2.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c3.metric("Precision", f"{metrics['precision']:.3f}")
    c4.metric("Recall", f"{metrics['recall']:.3f}")

    st.divider()

    # sidebar controls with presets
    st.sidebar.header("Lake Parameters")
    preset = st.sidebar.selectbox(
        "Scenario Preset",
        ["Custom", "Safe Lake", "Moderate Expansion", "Critical GLOF Risk"],
    )

    defaults = {
        "Custom": (16.0, 4.0, 0.01, 0.33, 0.00),
        "Safe Lake": (13.4, 1.5, -0.01, 0.25, 0.00),
        "Moderate Expansion": (19.0, 12.0, 0.08, 0.22, 0.05),
        "Critical GLOF Risk": (22.4, 30.4, 0.24, 0.17, 0.13),
    }[preset]

    mean_depth = st.sidebar.slider("mean_depth (m)", 6.5, 55.0, float(defaults[0]), 0.1)
    volume_dam = st.sidebar.slider("volume_dam (10^6 m^3)", 0.1, 100.0, float(defaults[1]), 0.5)
    volume_gap = st.sidebar.slider("volume_gap", -0.1, 5.0, float(defaults[2]), 0.01)
    uncertainty_frac = st.sidebar.slider("uncertainty_frac", 0.02, 1.3, float(defaults[3]), 0.01)
    expansion_frac = st.sidebar.slider("expansion_frac", -0.5, 1.0, float(defaults[4]), 0.01)

    input_df = pd.DataFrame([{
        "mean_depth": mean_depth,
        "volume_dam": volume_dam,
        "volume_gap": volume_gap,
        "uncertainty_frac": uncertainty_frac,
        "expansion_frac": expansion_frac,
    }])

    proba, pred = predict(input_df, model=model)
    prob_val = float(proba[0])
    is_critical = int(pred[0]) == 1

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Prediction Result")
        if is_critical:
            st.error(f"CRITICAL RISK — GLOF Probability: {prob_val:.1%}")
        else:
            st.success(f"LOW RISK — GLOF Probability: {prob_val:.1%}")

        st.progress(min(max(prob_val, 0.0), 1.0))
        st.caption(f"Classification threshold: {THRESHOLD:.2f}")

        # feature importance from fitted random forest
        rf = model.named_steps["rf"]
        imp_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=True)
        imp_fig = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation="h", marker_color="#0284c7"))
        imp_fig.update_layout(title="Model Feature Importances", height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(imp_fig, use_container_width=True)

    with right:
        st.subheader("Criticality Simulation")
        sim_fig = build_glof_animation(mean_depth, volume_dam, volume_gap, expansion_frac, prob_val)
        st.plotly_chart(sim_fig, use_container_width=True)
        if is_critical:
            st.warning("Simulation shows moraine crest erosion and rapid downstream surge reaching the settlement.")
        else:
            st.info("Simulation shows stable moraine freeboard with normal downstream baseflow.")


if __name__ == "__main__":
    main()

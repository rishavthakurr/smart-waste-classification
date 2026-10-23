"""Shared Streamlit UI helpers (the only module besides app.py/pages that imports streamlit)."""
from __future__ import annotations

import logging
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.database import add_prediction
from src.gradcam import GradCAMNotSupported, compute_gradcam, find_target_layer, overlay_heatmap
from src.model_loader import (ModelLoadError, ModelNotAvailableError, get_deployment_model,
                              input_size, list_models, load_model)
from src.predictor import PredictionError, PredictionResult, predict
from src.preprocessing import ImageValidationError, prepare_input, validate_image_bytes
from src.recommendations import SIMULATED_BINS, get_bin, get_recommendation
from src.utils import class_names, load_config, model_display_name, model_path, pretty

log = logging.getLogger("smart_waste")
GREEN, LIGHT, BLUE, GRAY = "#1B5E20", "#81C784", "#2F80ED", "#6B7280"
CATEGORY_COLORS = {c: b["color"] for c, b in SIMULATED_BINS.items()}          # same colours as the bin simulation
MODEL_COLORS = {"Custom CNN": GREEN, "MobileNetV2": LIGHT, "ResNet50": BLUE}
TRAIN_CMD = {"cnn": "train_cnn", "mobilenetv2": "train_mobilenet", "resnet50": "train_resnet"}

CSS = f"""
<style>
.block-container {{ padding-top: 4.2rem; max-width: 1200px; }}
h1, h2, h3 {{ color: {GREEN}; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid #DDE6DD; border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
[data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
[data-testid="stMetricValue"] div {{ white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.2; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}
.model-card {{ font-size: .95rem; }}
.badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; }}
.badge.ok {{ background:#E8F5E9; color:#1B5E20; }} .badge.warn {{ background:#FFF4D6; color:#8A6100; }}
.badge.off {{ background:#ECEFF1; color:#546E7A; }}
.hero {{ background: linear-gradient(135deg, {GREEN}, #2E7D32); color: #fff; border-radius: 16px;
    padding: 22px 26px; margin-bottom: 18px; }}
.hero h1 {{ color: #fff; margin: 0; font-size: 1.7rem; }}
.hero p {{ margin: 4px 0 0; opacity: .9; }}
.result-class {{ font-size: 1.9rem; font-weight: 700; color: {GREEN}; }}
.bin-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0; }}
.bin {{ flex: 1 1 90px; min-width: 90px; text-align: center; border-radius: 10px; padding: 10px 4px;
    color: #fff; font-size: .78rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,.5);
    opacity: .6; filter: grayscale(.55); border: 3px solid transparent; }}
.bin.active {{ opacity: 1; filter: none; border-color: #111; transform: translateY(-4px); }}
.empty {{ text-align:center; padding: 28px; border: 2px dashed #B7D3B9; border-radius: 16px; background:#F4FAF4; }}
</style>
"""


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    """One consistent look for every chart: white, light grid, legend underneath, no clutter."""
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=8, r=8, t=8, b=8),
                      font=dict(size=13, color="#1F2933"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, title_text=""))
    fig.update_xaxes(showgrid=False, linecolor="#CFD8CF")
    fig.update_yaxes(gridcolor="#E8EEE8", zeroline=False)
    return fig


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


def conf_fmt(p: float) -> str:
    return f"{p * 100:.{load_config()['ui']['confidence_decimals']}f}%"


@st.cache_resource(show_spinner="Loading model...")
def _cached_model(model_key: str, mtime: float):
    return load_model(model_key)


def get_model(model_key: str):
    return _cached_model(model_key, os.path.getmtime(model_path(model_key)))


def _names(items: list[str]) -> str:
    return ", ".join(f"**{n}**" for n in items)


def smoke_warning(models: dict | None = None) -> None:
    """One compact notice for models whose results must not be taken at face value."""
    models = models or list_models()
    smoke = [v["display_name"] for v in models.values() if v["available"] and v["smoke_test"]]
    nopre = [v["display_name"] for v in models.values()
             if v["available"] and v["pretrained"] is False and load_config()["models"][v["key"]]["transfer"]]
    if not (smoke or nopre):
        return
    parts = []
    if smoke:
        parts.append(f"{_names(smoke)} {'is a' if len(smoke) == 1 else 'are'} **smoke-test** "
                     f"model{'' if len(smoke) == 1 else 's'} (tiny subset, 2 epochs): predictions and metrics are not meaningful.")
    if nopre:
        parts.append(f"{_names(nopre)} {'was' if len(nopre) == 1 else 'were'} trained without ImageNet weights "
                     "(not true transfer learning).")
    st.warning(" ".join(parts) + " Train properly with `python -m training.train_*` (see README).")


def render_model_status() -> None:
    """Dashboard panel: which models are ready, and exactly how to train the missing ones."""
    models = list_models(); dep, _ = get_deployment_model()
    cols = st.columns(len(models))
    for col, (key, m) in zip(cols, models.items()):
        with col.container(border=True):
            if not m["available"]:
                badge, note = '<span class="badge off">Not trained</span>', f"Train it: `python -m training.{TRAIN_CMD[key]}`"
            elif m["smoke_test"]:
                badge, note = '<span class="badge warn">Smoke test only</span>', "Pipeline check - retrain for real use."
            else:
                badge, note = '<span class="badge ok">Ready</span>', ""
            star = " &nbsp;(deployment model)" if key == dep else ""
            st.markdown(f"**{m['display_name']}**{star}<br>{badge}", unsafe_allow_html=True)
            if note:
                st.caption(note)


def model_selector(prefix: str) -> str | None:
    """Radio of available models; unavailable ones are explained, never substituted."""
    models = list_models()
    avail = [k for k, v in models.items() if v["available"]]
    missing = [k for k, v in models.items() if not v["available"]]
    if not avail:
        st.warning("No trained model found. Train one first, e.g. "
                   "`python -m training.train_mobilenet`, then reload this page.")
        return None
    default, _ = get_deployment_model()
    idx = avail.index(default) if default in avail else 0
    choice = st.radio("Model", avail, index=idx, key=f"{prefix}_model", horizontal=True,
                      format_func=model_display_name)
    if missing:
        st.caption("Not available (train first): " + ", ".join(
            f"{model_display_name(k)} (`python -m training.{TRAIN_CMD[k]}`)" for k in missing))
    smoke_warning({choice: models[choice]})
    return choice


def prob_chart(result: PredictionResult) -> go.Figure:
    df = pd.DataFrame({"Category": [pretty(c) for c in result.probabilities],
                       "Probability": list(result.probabilities.values())}).sort_values("Probability")
    colors = [GREEN if c == pretty(result.predicted_class) else LIGHT for c in df["Category"]]
    fig = go.Figure(go.Bar(x=df["Probability"], y=df["Category"], orientation="h", marker_color=colors,
                           text=[conf_fmt(p) for p in df["Probability"]], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 1.12], tickformat=".0%"),
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def bin_simulation(category: str) -> None:
    items = "".join(
        f'<div class="bin{" active" if c == category else ""}" style="background:{b["color"]}">{b["bin"]}</div>'
        for c, b in SIMULATED_BINS.items())
    st.markdown(f'<div class="bin-row">{items}</div>', unsafe_allow_html=True)
    st.caption(f"Simulated decision: **{get_bin(category)['bin']}**. Software simulation only - "
               "no physical sorting mechanism is connected.")


def render_gradcam(image: Image.Image, model_key: str) -> None:
    try:
        model = get_model(model_key)
        layer = find_target_layer(model)
        scheme = load_config()["models"][model_key]["preprocessing"]
        batch = prepare_input(image, input_size(model), scheme)
        cam, idx = compute_gradcam(model, batch)
        if cam.max() <= 0:
            st.info("Grad-CAM produced an empty heatmap (the model's output is saturated for this image), "
                    "so there is nothing meaningful to overlay.")
            return
        colored, overlay = overlay_heatmap(image, cam)
    except GradCAMNotSupported as exc:
        st.info(str(exc)); return
    except Exception:
        log.exception("Grad-CAM failed")
        st.error("Grad-CAM could not be generated for this image/model."); return
    c1, c2 = st.columns(2)
    c1.image(image, caption="Original", width="stretch")
    c2.image(overlay, caption=f"Grad-CAM for '{pretty(class_names()[idx])}' (layer: {layer})", width="stretch")
    st.caption("Grad-CAM highlights the image regions that most increased the model's score for the "
               "predicted class (warmer = more influence). It explains the model's focus; "
               "it does **not** prove the prediction is correct.")


def render_result(result: PredictionResult, image: Image.Image, prefix: str) -> None:
    rec = get_recommendation(result.predicted_class)
    with st.container(border=True):
        st.caption("Predicted Category")
        st.markdown(f'<div class="result-class">{pretty(result.predicted_class)}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Confidence", conf_fmt(result.confidence))
        c2.metric("Model", model_display_name(result.model_key))
        if result.low_confidence:
            st.warning("Low confidence: the model is unsure. Consider a clearer photo or checking manually.")
        st.markdown(f"**Recommended Action:** {rec['action']}")
        st.caption(f"Tip: {rec['tip']}")
    st.caption("Confidence is the model's predicted probability for this class, not a guarantee of correctness.")
    st.plotly_chart(prob_chart(result), width="stretch", key=f"{prefix}_probs")
    top = " | ".join(f"{i}. {pretty(c)} {conf_fmt(p)}" for i, (c, p) in enumerate(result.top3, 1))
    st.markdown(f"**Top 3:** {top}")
    st.caption(f"Inference time (model only): {result.inference_ms:.1f} ms. {rec['disclaimer']}")
    st.subheader("Smart bin simulation")
    bin_simulation(result.predicted_class)
    with st.expander("Explainable AI: Grad-CAM"):
        if st.toggle("Generate Grad-CAM heatmap", key=f"{prefix}_gc"):
            render_gradcam(image, result.model_key)


def classify_section(prefix: str, data: bytes, filename: str) -> None:
    """Validate -> preview -> choose model -> classify -> show + persist result."""
    try:
        image = validate_image_bytes(data, filename)
    except ImageValidationError as exc:
        st.error(str(exc)); return
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(image, caption=f"{filename} ({image.width}x{image.height})", width="stretch")
        model_key = model_selector(prefix)
        go_clicked = st.button("Classify Waste", type="primary", width="stretch",
                               disabled=model_key is None, key=f"{prefix}_go")
    sig = (filename, len(data), model_key)
    if go_clicked and model_key:
        try:
            with st.spinner("Classifying..."):
                result = predict(image, model_key, get_model(model_key))
            add_prediction(filename, result.predicted_class, result.confidence, model_key, result.probabilities,
                           status="Low confidence" if result.low_confidence else "Confident")
            st.session_state[f"{prefix}_state"] = {"sig": sig, "result": result}
        except (ModelNotAvailableError, ModelLoadError, PredictionError) as exc:
            st.session_state.pop(f"{prefix}_state", None); right.error(str(exc))
        except Exception:
            log.exception("Unexpected classification error")
            st.session_state.pop(f"{prefix}_state", None)
            right.error("Something went wrong while classifying this image. Please try another image.")
    state = st.session_state.get(f"{prefix}_state")
    with right:
        if state and state["sig"] == sig:
            render_result(state["result"], image, prefix)
        elif model_key is None:
            st.info("Predictions are unavailable until a model is trained. Train at least one model "
                    "(the dataset must be prepared first), then reload this page:")
            st.code("python -m training.prepare_dataset --download\npython -m training.train_mobilenet\n"
                    "python -m training.evaluate_models", language="powershell")
        else:
            st.info("Choose a model and press **Classify Waste** to see the result.")

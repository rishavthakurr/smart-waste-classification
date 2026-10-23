import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import count_predictions, query_history
from src.model_loader import get_deployment_model, list_models
from src.ui import (CATEGORY_COLORS, GREEN, LIGHT, MODEL_COLORS, conf_fmt, empty_state, page_header,
                    render_model_status, smoke_warning, style_fig)
from src.utils import class_names, model_display_name, pretty, read_json, results_dir

page_header("Dashboard", "Live overview of classification activity and model status.")
models = list_models()
smoke_warning(models)
total, df = count_predictions(), query_history()
dep_key, dep_status = get_deployment_model()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Images classified", total)
c2.metric("Waste classes supported", len(class_names()))
c3.metric("Most predicted category", pretty(df["predicted_class"].mode().iloc[0]) if total else "-")
c4.metric("Deployment model", model_display_name(dep_key) if dep_key else "None", help=dep_status)
if not dep_key:
    st.caption(f"Deployment model: {dep_status}.")
st.subheader("Models")
render_model_status()

if total == 0:
    empty_state("No predictions yet",
                "Open <b>Classify Waste</b> (or <b>Live Detection</b>), upload an image and press "
                "<b>Classify Waste</b>. Your results and charts will appear here.")
else:
    a, b = st.columns(2)
    dist = df["predicted_class"].value_counts().rename_axis("category").reset_index(name="count")
    dist["category"] = dist["category"].map(pretty)
    with a.container(border=True):
        st.markdown("**Waste category distribution**")
        fig = px.pie(dist, names="category", values="count", hole=.55, color="category",
                     color_discrete_map={pretty(c): col for c, col in CATEGORY_COLORS.items()})
        fig.update_traces(textinfo="percent", sort=False)
        st.plotly_chart(style_fig(fig, 320), width="stretch")
    with b.container(border=True):
        st.markdown("**Predictions by category and model**")
        g = df.assign(category=df["predicted_class"].map(pretty), model=df["model"].map(model_display_name)) \
              .groupby(["category", "model"]).size().reset_index(name="count")
        fig = px.bar(g, x="category", y="count", color="model", barmode="group",
                     color_discrete_map=MODEL_COLORS, labels={"category": "", "count": "Predictions", "model": ""})
        fig.update_yaxes(dtick=1 if g["count"].max() < 8 else None)
        st.plotly_chart(style_fig(fig, 320), width="stretch")
    with st.container(border=True):
        st.markdown("**Recent classification activity**")
        daily = df.assign(day=df["timestamp"].dt.strftime("%d %b %Y")).groupby("day", sort=False).size() \
                  .iloc[::-1].reset_index(name="classifications")          # oldest -> newest, categorical axis
        fig = px.bar(daily, x="day", y="classifications", color_discrete_sequence=[GREEN],
                     labels={"day": "Day (UTC)", "classifications": "Classifications"})
        fig.update_xaxes(type="category"); fig.update_yaxes(dtick=1 if daily["classifications"].max() < 8 else None)
        st.plotly_chart(style_fig(fig, 280), width="stretch")
    st.subheader("Recent predictions")
    recent = df.head(10)
    st.dataframe(pd.DataFrame({
        "Timestamp": recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Predicted class": recent["predicted_class"].map(pretty),
        "Confidence": recent["confidence"].map(conf_fmt),
        "Image filename": recent["image_filename"], "Status": recent["status"]}),
        width="stretch", hide_index=True)

st.subheader("Model performance comparison")
cmp_path = results_dir() / "model_comparison.csv"
if cmp_path.exists():
    cmp = pd.read_csv(cmp_path)
    if cmp["smoke_test"].any():
        st.warning("Some of these results come from smoke-test models and are not meaningful.")
    names = {"test_accuracy": "Accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}
    long = cmp.melt(id_vars="model", value_vars=list(names), var_name="metric", value_name="score")
    long["metric"] = long["metric"].map(names)
    fig = px.bar(long, x="model", y="score", color="metric", barmode="group", range_y=[0, 1],
                 color_discrete_sequence=[GREEN, LIGHT, "#2F80ED"], labels={"model": "", "score": "Score", "metric": ""})
    st.plotly_chart(style_fig(fig, 320), width="stretch")
else:
    st.info("Evaluation pending. Train models, then run `python -m training.evaluate_models`.")

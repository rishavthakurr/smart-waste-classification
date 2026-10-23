import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import GREEN, LIGHT, MODEL_COLORS, page_header, smoke_warning, style_fig
from src.model_loader import list_models
from src.utils import pretty, read_json, results_dir

page_header("Model Performance", "Test-set evaluation produced by training/evaluate_models.py.")
res = results_dir()
cmp_path, pc_path = res / "model_comparison.csv", res / "per_class_metrics.csv"
if not cmp_path.exists():
    st.info("Training and evaluation are pending - no results exist yet.")
    st.code("python -m training.prepare_dataset --download\npython -m training.train_cnn\n"
            "python -m training.train_mobilenet\npython -m training.train_resnet\n"
            "python -m training.evaluate_models", language="powershell")
    st.stop()

cmp, per = pd.read_csv(cmp_path), pd.read_csv(pc_path)
if cmp["smoke_test"].any():
    st.warning("These results include **smoke-test** models (tiny subset, 2 epochs). They only prove the "
               "pipeline runs; do not report them.")
rec = read_json(res / "recommended_model.json")
if isinstance(rec, dict):
    st.success(f"Recommended deployment model: **{rec['model']}** (rule: {rec['rule']}).")

st.subheader("Model comparison")
show = cmp[["model", "test_accuracy", "macro_f1", "weighted_f1", "inference_ms_per_image", "model_size_mb"]]
st.dataframe(show.rename(columns={"model": "Model", "test_accuracy": "Test accuracy", "macro_f1": "Macro F1",
             "weighted_f1": "Weighted F1", "inference_ms_per_image": "Inference (ms/img)",
             "model_size_mb": "Size (MB)"}), hide_index=True, width="stretch",
             column_config={"Test accuracy": st.column_config.NumberColumn(format="%.4f"),
                            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
                            "Weighted F1": st.column_config.NumberColumn(format="%.4f"),
                            "Inference (ms/img)": st.column_config.NumberColumn(format="%.1f"),
                            "Size (MB)": st.column_config.NumberColumn(format="%.1f")})
st.caption("Inference time = median of 100 single-image forward passes after 10 warm-ups; excludes image loading. "
           "Measured on the machine that ran the evaluation.")
a, b = st.columns(2)
names = {"test_accuracy": "Accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}
long = cmp.melt(id_vars="model", value_vars=list(names), var_name="metric", value_name="score")
long["metric"] = long["metric"].map(names)
with a.container(border=True):
    st.markdown("**Accuracy and F1 (test set)**")
    st.plotly_chart(style_fig(px.bar(long, x="model", y="score", color="metric", barmode="group", range_y=[0, 1],
                    color_discrete_sequence=[GREEN, LIGHT, "#2F80ED"], labels={"model": "", "score": "Score", "metric": ""})),
                    width="stretch")
with b.container(border=True):
    st.markdown("**Per-class F1 (test set)**")
    per_view = per.assign(**{"class": per["class"].map(pretty)})
    st.plotly_chart(style_fig(px.bar(per_view, x="class", y="f1", color="model", barmode="group", range_y=[0, 1],
                    color_discrete_map=MODEL_COLORS, labels={"class": "", "f1": "F1-score", "model": ""})),
                    width="stretch")

st.subheader("Inspect a model")
choice = st.selectbox("Model", list(cmp["model_key"]), format_func=lambda k: cmp.set_index("model_key").loc[k, "model"])
row = cmp.set_index("model_key").loc[choice]
meta = read_json(res / choice / "meta.json") or {}
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test accuracy", f"{row['test_accuracy']:.4f}"); m2.metric("Macro F1", f"{row['macro_f1']:.4f}")
m3.metric("Parameters", f"{int(row['parameters']):,}"); m4.metric("Epochs run", meta.get("epochs_run", "-"))
smoke_warning({choice: list_models()[choice]})

t1, t2, t3 = st.tabs(["Per-class metrics", "Confusion matrix", "Training history"])
with t1:
    pc = per[per["model_key"] == choice][["class", "precision", "recall", "f1", "support"]].copy()
    pc["class"] = pc["class"].map(pretty)
    st.dataframe(pc.rename(columns={"class": "Category", "precision": "Precision", "recall": "Recall",
                                    "f1": "F1-score", "support": "Test images"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("Precision", "Recall", "F1-score")})
with t2:
    cm_path = res / choice / "confusion_matrix.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0)
        cm.index, cm.columns = [pretty(c) for c in cm.index], [pretty(c) for c in cm.columns]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Greens", labels=dict(x="Predicted", y="True", color="Count"))
        st.plotly_chart(style_fig(fig, 460), width="stretch")
        st.caption("Rows are the true category, columns the model's prediction; the diagonal shows correct predictions.")
with t3:
    h_path = res / choice / "history.csv"
    if h_path.exists():
        h = pd.read_csv(h_path)
        lab = {"accuracy": "Train", "val_accuracy": "Validation", "loss": "Train", "val_loss": "Validation"}
        for cols, ttl in ((["accuracy", "val_accuracy"], "Accuracy"), (["loss", "val_loss"], "Loss")):
            long_h = h.melt(id_vars=["epoch"], value_vars=cols, var_name="series", value_name=ttl)
            long_h["series"] = long_h["series"].map(lab)
            fig = px.line(long_h, x="epoch", y=ttl, color="series", markers=True,
                          color_discrete_map={"Train": GREEN, "Validation": "#2F80ED"}, labels={"series": "", "epoch": "Epoch"})
            fig.update_xaxes(dtick=1 if len(h) <= 15 else None)
            st.markdown(f"**{ttl}**"); st.plotly_chart(style_fig(fig, 280), width="stretch")
    else:
        st.info("No training history found for this model.")

rep = read_json(res / "dataset_report.json")
if isinstance(rep, dict):
    with st.expander("Dataset report"):
        st.write(f"Usable images: **{rep['usable_images']}** (excluded {rep['excluded_images']}). "
                 f"Imbalance ratio {rep['imbalance_ratio_max_over_min']}:1.")
        st.json({"images_per_class": rep["images_per_class"], "split_counts": rep["split_counts"]})
        for img in ("class_distribution.png", "split_distribution.png"):
            if (res / "dataset" / img).exists():
                st.image(str(res / "dataset" / img))

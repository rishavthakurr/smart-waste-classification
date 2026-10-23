import streamlit as st

from src.database import clear_history, count_predictions, export_csv, query_history
from src.ui import conf_fmt, empty_state, page_header
from src.utils import class_names, load_config, model_display_name, pretty

page_header("Classification History", "Every prediction made in this app, stored locally in SQLite.")
if count_predictions() == 0:
    empty_state("No history yet", "Classify an image on the <b>Classify Waste</b> page and it will appear here.")
    st.stop()

c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1])
cat = c1.selectbox("Category", ["All"] + class_names(), format_func=lambda c: c if c == "All" else pretty(c))
mod = c2.selectbox("Model", ["All"] + list(load_config()["models"]),
                   format_func=lambda m: m if m == "All" else model_display_name(m))
search = c3.text_input("Search filename")
order = c4.selectbox("Sort by time", ["Newest first", "Oldest first"])

df = query_history(None if cat == "All" else cat, None if mod == "All" else mod, search or None,
                   newest_first=order == "Newest first")
m1, m2, m3 = st.columns(3)
m1.metric("Matching records", len(df))
m2.metric("Average confidence", conf_fmt(df["confidence"].mean()) if len(df) else "-")
m3.metric("Low-confidence predictions", int((df["status"] == "Low confidence").sum()) if len(df) else 0)

if df.empty:
    st.info("No records match the current filters.")
else:
    view = df.assign(timestamp=df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                     predicted_class=df["predicted_class"].map(pretty),
                     model=df["model"].map(model_display_name),
                     confidence=df["confidence"] * 100)
    st.dataframe(view[["id", "timestamp", "image_filename", "predicted_class", "confidence", "model", "status"]],
                 width="stretch", hide_index=True,
                 column_config={"id": st.column_config.NumberColumn("#", width="small"),
                                "timestamp": st.column_config.TextColumn("Time", width="medium"),
                                "image_filename": st.column_config.TextColumn("Image", width="medium"),
                                "predicted_class": st.column_config.TextColumn("Category", width="small"),
                                "confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%",
                                                                              min_value=0, max_value=100),
                                "model": st.column_config.TextColumn("Model", width="small"),
                                "status": st.column_config.TextColumn("Status", width="medium")})
    st.download_button("Download CSV", export_csv(df), "classification_history.csv", "text/csv")

with st.expander("Danger zone: clear history"):
    st.warning("This permanently deletes every stored prediction.")
    ok = st.checkbox("I understand this cannot be undone")
    if st.button("Clear all history", disabled=not ok):
        st.toast(f"Deleted {clear_history()} records.", icon=":material/delete:")
        st.rerun()

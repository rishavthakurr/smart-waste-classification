import streamlit as st

from src.ui import page_header
from src.utils import load_config, pretty, read_json, results_dir

cfg = load_config()
page_header("About the Project", cfg["project"]["title"])
st.markdown("""
**What it does.** Classifies a waste photo into one of six categories with deep learning, explains the
decision with Grad-CAM, suggests a disposal action, and records every prediction in a local SQLite database.

**Implemented features**
- Three models: custom CNN, MobileNetV2 and ResNet50 (transfer learning), trained and evaluated on one held-out test split
- Upload and webcam-capture classification with real softmax confidence and top-3 probabilities
- Grad-CAM heatmaps from the actual trained model
- Dashboard, history (filter, search, CSV export), model-performance page
- Software-only smart-bin simulation

**Future enhancements (not implemented)**: a physical sorting mechanism (microcontroller + servo),
continuous video detection, multi-item detection in one image, a larger and more varied dataset.
""")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Classes and dataset mapping")
    st.table({"Dataset label": list(cfg["dataset"]["label_map"]),
              "Project class": [pretty(v) for v in cfg["dataset"]["label_map"].values()]})
with c2:
    st.subheader("Dataset")
    st.markdown("TrashNet by Gary Thung and Mindy Yang (Stanford CS229, 2016), MIT licence. "
                "[github.com/garythung/trashnet](https://github.com/garythung/trashnet). "
                "Please cite the repository when you use it.")
    rep = read_json(results_dir() / "dataset_report.json")
    if isinstance(rep, dict):
        st.caption(f"Prepared locally: {rep['usable_images']} usable images.")
st.subheader("Known limitations")
st.markdown("""
- The dataset is small (about 2.5k images) and photographed on a white background, so accuracy on cluttered real-world photos will be lower than test-set accuracy.
- The "General Waste" class is the smallest (about 5% of images); class weights reduce but do not remove the imbalance.
- One item per image is assumed. Disposal advice is general and depends on local rules.
- Confidence scores are model probabilities, not guarantees. Grad-CAM shows model focus, not correctness.
""")

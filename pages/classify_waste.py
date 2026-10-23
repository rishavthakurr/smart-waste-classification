import streamlit as st

from src.ui import classify_section, page_header

page_header("Classify Waste", "Upload a photo of a waste item to identify its category and get disposal guidance.")
ss = st.session_state
ss.setdefault("upload_counter", 0)
key = f"uploader_{ss['upload_counter']}"
file = st.file_uploader("Upload a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"], key=key,
                        help="Max 10 MB. Re-uploading replaces the current image.")
if file is None:
    st.info("Upload an image to begin. Tip: a single item on a plain background works best "
            "(the training dataset used white backgrounds).")
else:
    classify_section("classify", file.getvalue(), file.name)
    if st.button("Classify another image"):
        ss.pop("classify_state", None)
        ss["upload_counter"] += 1
        st.rerun()

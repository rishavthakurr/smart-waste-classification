import streamlit as st

from src.ui import classify_section, page_header

page_header("Live Detection", "Capture a frame with your webcam, then classify it.")
ss = st.session_state
ss.setdefault("cam_counter", 0)
tab_cam, tab_up = st.tabs(["Webcam capture", "Upload instead"])

with tab_cam:
    st.caption("Your browser will ask for camera permission. Webcam access needs HTTPS or localhost "
               "and may be unavailable in some browsers/deployments. If it fails, use the Upload tab.")
    shot = st.camera_input("Take a photo of the waste item", key=f"cam_{ss['cam_counter']}")
    if shot is not None:
        classify_section("live", shot.getvalue(), "webcam_capture.png")
        if st.button("Capture another image"):
            ss.pop("live_state", None)
            ss["cam_counter"] += 1
            st.rerun()

with tab_up:
    up = st.file_uploader("Upload an image (fallback when no webcam)", type=["jpg", "jpeg", "png"], key="live_upload")
    if up is not None:
        classify_section("live_up", up.getvalue(), up.name)

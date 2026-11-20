"""Streamlit Community Cloud portfolio adapter for plant-disease-detector.

This does not replace the original implementation.
"""
from pathlib import Path

import streamlit as st

TITLE = "Plant Leaf Classifier — Starter"
ICON = "🌿"
DESC = "A basic RandomForest image classification pipeline designed for locally supplied, labeled plant images."
CATEGORY = "Experimental ML starter"
NOTICE = "No trained classifier or verified disease diagnostic performance is included. Uploaded sample photos in this Streamlit app are displayed locally in the session, not classified."
GITHUB = "https://github.com/mohsinattasial/plant-disease-detector"
VERCEL = "https://mohsin-plant-classifier.vercel.app"
LOCAL_COMMAND = "python train.py --data data --model model.joblib"
FEATURES = ["Image feature extraction","Held-out evaluation script","Training and CLI prediction"]

st.set_page_config(page_title=TITLE + " | Mohsin Atta", page_icon=ICON, layout="wide")
st.caption("Mohsin Atta · GitHub Portfolio · Streamlit Cloud edition")
st.title(ICON + " " + TITLE)
st.write(DESC)
st.caption(CATEGORY)

left, right = st.columns(2)
with left:
    st.link_button("GitHub source code", GITHUB, use_container_width=True)
with right:
    st.link_button("Existing browser version", VERCEL, use_container_width=True)

st.info(NOTICE)
with st.expander("Original project and features", expanded=False):
    for feature in FEATURES:
        st.write("• " + feature)
    st.code(LOCAL_COMMAND, language="bash")

st.subheader("Leaf photo preview — no diagnosis")
upload = st.file_uploader("Choose a plant leaf photo (optional)", type=["jpg", "jpeg", "png", "webp"])
if upload:
    st.image(upload, caption="Preview only — no prediction performed", use_container_width=True)
st.markdown("### Prepare a local training dataset")
st.code("data/healthy/example1.jpg\ndata/rust/example2.jpg", language="text")
st.warning("No trained model is shipped with the project. Run train.py on evaluated labeled data before considering predictions.")

st.divider()
st.caption("© 2026 Mohsin Atta · Educational portfolio · MIT-licensed original code where applicable")

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Healthcare Risk Prioritization", layout="wide")
st.title("FAIR-style Healthcare Risk Prioritization")

artifact_dir = st.sidebar.text_input("Artifact directory", value="artifacts/uci")
artifact_path = Path(artifact_dir)

metrics_file = artifact_path / "metrics.json"
queue_file = artifact_path / "ranked_queue.csv"
calibration_file = artifact_path / "calibration_curve.png"
shap_file = artifact_path / "shap_summary.png"

if not queue_file.exists() or not metrics_file.exists():
    st.info("Run training first. Expect metrics.json + ranked_queue.csv in the artifact directory.")
    st.stop()

with open(metrics_file, "r", encoding="utf-8") as f:
    metrics = json.load(f)

c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC", f"{metrics.get('auc', 0):.3f}")
c2.metric("PR AUC", f"{metrics.get('pr_auc', 0):.3f}")
c3.metric("Brier", f"{metrics.get('brier', 0):.3f}")
c4.metric("Mean Risk", f"{metrics.get('mean_risk', 0):.3f}")

queue = pd.read_csv(queue_file)

st.subheader("Ranked Intervention Queue")
max_rows = st.slider("Rows to display", min_value=10, max_value=min(500, len(queue)), value=min(50, len(queue)))
st.dataframe(queue.head(max_rows), use_container_width=True)

st.download_button(
    label="Download ranked queue CSV",
    data=queue.to_csv(index=False).encode("utf-8"),
    file_name="ranked_queue.csv",
    mime="text/csv",
)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Calibration")
    if calibration_file.exists():
        st.image(str(calibration_file))
    else:
        st.caption("No calibration image found.")

with col_right:
    st.subheader("SHAP Summary")
    if shap_file.exists():
        st.image(str(shap_file))
    else:
        st.caption("No SHAP summary image found.")

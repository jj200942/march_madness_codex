# FAIR-style Healthcare Risk Prioritization

This project provides a practical starter for **healthcare risk prioritization** with:

- adverse event probability modeling
- impact modeling (cost / health outcome proxy)
- FAIR-style risk scoring (`risk = probability × impact`)
- calibration curves
- SHAP interpretability + clinician-friendly explanations
- causal uplift modeling (T-learner treatment effect estimates)
- ranked intervention queue and a Streamlit decision dashboard

It supports two data pathways:

1. **UCI Diabetes Readmission** for fast iteration.
2. **MIMIC-IV feature pipeline** scaffold for production-grade extension.

---

## 1) Quick Start (UCI workflow)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Add data

Download UCI Diabetes Readmission dataset and place it at:

```text
./data/uci/diabetic_data.csv
```

Source:
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

### Train + evaluate

```bash
python src/train_uci.py \
  --data-path data/uci/diabetic_data.csv \
  --output-dir artifacts/uci
```

This produces:

- model metrics (`metrics.json`)
- ranked intervention queue (`ranked_queue.csv`)
- calibration curve (`calibration_curve.png`)
- SHAP summary (`shap_summary.png`)
- optional uplift scores (`uplift_scores.csv`, when treatment column exists)

### Run decision dashboard

```bash
streamlit run src/streamlit_app.py
```

Set the sidebar path to `artifacts/uci`.

---

## 2) Real MIMIC-IV pipeline scaffold

Run:

```bash
python src/mimic_iv_pipeline.py \
  --mimic-dir /path/to/mimiciv \
  --output-path data/mimic/features.parquet
```

Expected MIMIC-IV files (CSV or gz CSV):

- `hosp/admissions`
- `hosp/patients`
- `hosp/diagnoses_icd`
- `hosp/d_icd_diagnoses`

The script creates a patient/admission-level feature table with:

- demographics and admission context
- prior utilization and length-of-stay signals
- one-hot diagnosis chapter proxies
- readmission-like proxy label (30-day revisit)
- impact proxy (`los_days`)

After feature generation, you can route the produced file into the same training flow as UCI.

---

## FAIR-style framing used in this repo

For each patient/intervention candidate:

- **Threat event probability**: `P(adverse_event)` from classifier.
- **Loss magnitude / impact**: predicted continuous impact from regressor.
- **Risk**: `risk_score = P(adverse_event) × expected_impact`.

This yields a clinically actionable **ranked intervention queue**.

---

## Notes

- This is a starter implementation, not medical advice.
- For real deployment, add strict cohort definitions, leakage checks, temporal validation, bias/fairness audits, and governance signoff.

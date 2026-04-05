from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from risk_pipeline import train_fair_risk_model


def load_uci_diabetes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Common cleaning for UCI Diabetes Readmission dataset.
    df = df.replace("?", np.nan)

    # Binary adverse-event label: readmitted within <30 days.
    df["adverse_event"] = (df["readmitted"].astype(str).str.upper() == "<30").astype(int)

    # Impact proxy: time in hospital (days).
    df["impact_proxy"] = pd.to_numeric(df["time_in_hospital"], errors="coerce")

    # Example treatment proxy: change in diabetes medications.
    if "change" in df.columns:
        df["treatment_intensified"] = (df["change"].astype(str).str.upper() == "CH").astype(int)

    # Remove identifier-like fields that can leak or add noise.
    drop_cols = [
        c
        for c in ["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code", "medical_specialty"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/uci")
    args = parser.parse_args()

    data = load_uci_diabetes(args.data_path)
    treatment_col = "treatment_intensified" if "treatment_intensified" in data.columns else None

    train_fair_risk_model(
        df=data,
        adverse_event_col="adverse_event",
        impact_col="impact_proxy",
        output_dir=args.output_dir,
        treatment_col=treatment_col,
        intervention_cost=0.2,
    )

    print(f"Done. Outputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

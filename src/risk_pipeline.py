from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RiskArtifacts:
    metrics: dict
    ranked_queue: pd.DataFrame


def _build_preprocessor(df: pd.DataFrame, target_cols: list[str]) -> ColumnTransformer:
    feature_cols = [c for c in df.columns if c not in target_cols]
    categorical_cols = [c for c in feature_cols if df[c].dtype == "object"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def train_fair_risk_model(
    df: pd.DataFrame,
    adverse_event_col: str,
    impact_col: str,
    output_dir: str,
    treatment_col: Optional[str] = None,
    intervention_cost: float = 0.0,
    random_state: int = 42,
) -> RiskArtifacts:
    """Train a FAIR-style risk stack and save outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df.dropna(subset=[adverse_event_col, impact_col])

    y_event = df[adverse_event_col].astype(int)
    y_impact = df[impact_col].astype(float)

    target_cols = [adverse_event_col, impact_col]
    if treatment_col:
        target_cols.append(treatment_col)

    X = df.drop(columns=target_cols)

    X_train, X_test, y_event_train, y_event_test, y_impact_train, y_impact_test = train_test_split(
        X,
        y_event,
        y_impact,
        test_size=0.25,
        random_state=random_state,
        stratify=y_event,
    )

    preprocessor = _build_preprocessor(df.drop(columns=[]), target_cols=target_cols)

    event_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=random_state)),
        ]
    )
    impact_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("reg", GradientBoostingRegressor(random_state=random_state)),
        ]
    )

    event_model.fit(X_train, y_event_train)
    impact_model.fit(X_train, y_impact_train)

    p_event = event_model.predict_proba(X_test)[:, 1]
    p_event_full = event_model.predict_proba(X)[:, 1]

    impact_pred = np.maximum(impact_model.predict(X_test), 0.0)
    impact_pred_full = np.maximum(impact_model.predict(X), 0.0)

    risk_score = p_event_full * impact_pred_full

    metrics = {
        "auc": float(roc_auc_score(y_event_test, p_event)),
        "pr_auc": float(average_precision_score(y_event_test, p_event)),
        "brier": float(brier_score_loss(y_event_test, p_event)),
        "impact_mae": float(np.mean(np.abs(impact_pred - y_impact_test))),
        "mean_risk": float(np.mean(risk_score)),
    }

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_event_test, p_event, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "calibration_curve.png", dpi=150)
    plt.close()

    # SHAP explainability on event model
    transformed = event_model.named_steps["preprocessor"].transform(X_test)
    feature_names = event_model.named_steps["preprocessor"].get_feature_names_out()
    explainer = shap.TreeExplainer(event_model.named_steps["clf"])
    shap_values = explainer.shap_values(transformed)

    plt.figure()
    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    queue = df.copy()
    queue["p_adverse_event"] = p_event_full
    queue["expected_impact"] = impact_pred_full
    queue["risk_score"] = risk_score

    if treatment_col and treatment_col in df.columns:
        uplift = estimate_t_learner_uplift(
            data=df,
            treatment_col=treatment_col,
            outcome_col=adverse_event_col,
            random_state=random_state,
        )
        queue = queue.join(uplift)
        queue["expected_net_benefit"] = (
            queue["uplift_abs_risk_reduction"].fillna(0.0) * queue["expected_impact"] - intervention_cost
        )
        uplift.to_csv(output / "uplift_scores.csv", index=False)

    queue = queue.sort_values("risk_score", ascending=False)
    queue.to_csv(output / "ranked_queue.csv", index=False)

    with open(output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return RiskArtifacts(metrics=metrics, ranked_queue=queue)


def estimate_t_learner_uplift(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simple T-learner uplift: E[Y|T=1,X] - E[Y|T=0,X]."""

    d = data.copy()
    d = d.dropna(subset=[treatment_col, outcome_col])

    y = d[outcome_col].astype(int)
    t = d[treatment_col].astype(int)
    X = d.drop(columns=[outcome_col, treatment_col])

    preprocessor = _build_preprocessor(d, target_cols=[outcome_col, treatment_col])

    treated_idx = t == 1
    control_idx = t == 0

    if treated_idx.sum() < 50 or control_idx.sum() < 50:
        return pd.DataFrame(
            {
                "uplift_abs_risk_reduction": np.full(len(d), np.nan),
                "uplift_segment": ["insufficient_data"] * len(d),
            },
            index=d.index,
        )

    treated_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=random_state)),
        ]
    )
    control_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=random_state + 1)),
        ]
    )

    treated_model.fit(X[treated_idx], y[treated_idx])
    control_model.fit(X[control_idx], y[control_idx])

    p_treated = treated_model.predict_proba(X)[:, 1]
    p_control = control_model.predict_proba(X)[:, 1]

    uplift = p_control - p_treated  # positive => expected risk reduction when treated

    uplift_segment = pd.qcut(uplift, q=5, labels=["lowest", "low", "medium", "high", "highest"])
    return pd.DataFrame(
        {
            "uplift_abs_risk_reduction": uplift,
            "uplift_segment": uplift_segment.astype(str),
        },
        index=d.index,
    )

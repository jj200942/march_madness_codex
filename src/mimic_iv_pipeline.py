from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def build_mimic_features(mimic_dir: str, output_path: str) -> None:
    """Create an admission-level feature table from core MIMIC-IV hosp tables."""

    root = Path(mimic_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    def resolve_csv(prefix: str) -> str:
        csv_path = root / f"{prefix}.csv"
        gz_path = root / f"{prefix}.csv.gz"
        if csv_path.exists():
            return str(csv_path)
        if gz_path.exists():
            return str(gz_path)
        raise FileNotFoundError(f"Missing {prefix}.csv or {prefix}.csv.gz under {root}")

    admissions_path = resolve_csv("hosp/admissions")
    patients_path = resolve_csv("hosp/patients")
    diagnoses_path = resolve_csv("hosp/diagnoses_icd")
    d_icd_path = resolve_csv("hosp/d_icd_diagnoses")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW admissions AS
        SELECT
            subject_id,
            hadm_id,
            admittime,
            dischtime,
            admission_type,
            insurance,
            language,
            marital_status,
            race,
            hospital_expire_flag
        FROM read_csv_auto('{admissions_path}', ignore_errors=true);
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW patients AS
        SELECT subject_id, gender, anchor_age, anchor_year
        FROM read_csv_auto('{patients_path}', ignore_errors=true);
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW diagnoses AS
        SELECT subject_id, hadm_id, icd_code, icd_version
        FROM read_csv_auto('{diagnoses_path}', ignore_errors=true);
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW d_icd AS
        SELECT icd_code, icd_version, long_title
        FROM read_csv_auto('{d_icd_path}', ignore_errors=true);
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE feature_table AS
        WITH base AS (
            SELECT
                a.subject_id,
                a.hadm_id,
                CAST(a.admittime AS TIMESTAMP) AS admittime,
                CAST(a.dischtime AS TIMESTAMP) AS dischtime,
                DATE_DIFF('day', CAST(a.admittime AS TIMESTAMP), CAST(a.dischtime AS TIMESTAMP)) AS los_days,
                a.admission_type,
                a.insurance,
                a.language,
                a.marital_status,
                a.race,
                a.hospital_expire_flag,
                p.gender,
                p.anchor_age
            FROM admissions a
            LEFT JOIN patients p USING(subject_id)
        ),
        dx AS (
            SELECT
                d.subject_id,
                d.hadm_id,
                COUNT(*) AS dx_count,
                SUM(CASE WHEN d.icd_version = 10 AND LEFT(d.icd_code, 1) IN ('I', 'J') THEN 1 ELSE 0 END) AS cardio_resp_dx,
                SUM(CASE WHEN d.icd_version = 10 AND LEFT(d.icd_code, 1) IN ('E') THEN 1 ELSE 0 END) AS endocrine_dx
            FROM diagnoses d
            GROUP BY 1, 2
        ),
        revisits AS (
            SELECT
                subject_id,
                hadm_id,
                CASE
                    WHEN LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime)
                         <= dischtime + INTERVAL 30 DAY THEN 1
                    ELSE 0
                END AS adverse_event_30d
            FROM base
        )
        SELECT
            b.subject_id,
            b.hadm_id,
            b.admittime,
            b.dischtime,
            COALESCE(b.los_days, 0) AS impact_proxy,
            b.admission_type,
            b.insurance,
            b.language,
            b.marital_status,
            b.race,
            b.hospital_expire_flag,
            b.gender,
            b.anchor_age,
            COALESCE(dx.dx_count, 0) AS dx_count,
            COALESCE(dx.cardio_resp_dx, 0) AS cardio_resp_dx,
            COALESCE(dx.endocrine_dx, 0) AS endocrine_dx,
            rv.adverse_event_30d AS adverse_event
        FROM base b
        LEFT JOIN dx USING(subject_id, hadm_id)
        LEFT JOIN revisits rv USING(subject_id, hadm_id);
        """
    )

    con.execute(f"COPY feature_table TO '{out}' (FORMAT PARQUET);")
    print(f"Wrote features to: {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", required=True)
    parser.add_argument("--output-path", default="data/mimic/features.parquet")
    args = parser.parse_args()

    build_mimic_features(args.mimic_dir, args.output_path)


if __name__ == "__main__":
    main()

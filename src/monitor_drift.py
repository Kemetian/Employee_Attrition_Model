# src/monitor_drift.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import numpy as np
import pandas as pd

# Import valid YAML config loader from your preprocessing utility
from src.preprocessing import load_config

# Import Evidently core Report engine and the Data Drift metric preset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

@dataclass(frozen=True)
class DriftReport:
    """Results of an Evidently-powered drift check."""
    feature_results: pd.DataFrame
    drifted_features: tuple[str, ...]
    drift_detected: bool

def monitor_drift(
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    *,
    alpha: float = 0.05,
    features: list[str] | None = None,
) -> DriftReport:
    """
    Test shared feature distributions for statistically significant drift using Evidently AI.
    Generates interactive HTML assets and exposes structural dataset-level evaluations.
    """
    # ── Load Configuration and Data Sources Dynamically ──
    if reference is None or current is None:
        config = load_config("configs/model_config.yaml")
        data_path = config["data"]["path"]

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at configured path: {data_path}")

        print(f"Loading master dataset from: {data_path}")
        df = pd.read_csv(data_path, on_bad_lines='warn')

        # Split the dataset 50/50 to simulate Reference (Historical) vs Current (Production Logs)
        midpoint = len(df) // 2
        if reference is None:
            reference = df.iloc[:midpoint].copy()
        if current is None:
            current = df.iloc[midpoint:].copy()

    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise TypeError("reference and current must be pandas DataFrames")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")

    # Align features to evaluate
    selected = list(features) if features is not None else list(
        reference.columns.intersection(current.columns)
    )

    missing = [feature for feature in selected if feature not in reference.columns or feature not in current.columns]
    if missing:
        raise ValueError(f"Features missing from one or both data frames: {missing}")

    # Build and compute an automated structured report using Evidently presets
    drift_report_engine = Report(metrics=[DataDriftPreset(columns=selected)])
    drift_report_engine.run(reference_data=reference, current_data=current)

    # Extract the dictionary structure to map back to DriftReport fields
    raw_dict = drift_report_engine.as_dict()
    drift_metrics = raw_dict["metrics"][0]["result"]

    # Check if dataset-level drift threshold has been bypassed
    dataset_drift_detected = drift_metrics["dataset_drift"]

    # Map feature-level metrics out to a clean Pandas DataFrame
    rows: list[dict[str, Any]] = []
    drifted_features_list = []

    for feature_name, info in drift_metrics["drift_by_columns"].items():
        is_drifted = info["drift_detected"]
        if is_drifted:
            drifted_features_list.append(feature_name)

        rows.append({
            "feature": feature_name,
            "test": info["stattest_name"],
            "statistic": info["drift_score"],
            "p_value": info.get("p_value", np.nan),  # Safe handling for metrics without p-values
            "drift_detected": bool(is_drifted),
        })

    results_df = pd.DataFrame(
        rows,
        columns=["feature", "test", "statistic", "p_value", "drift_detected"],
    )

    # Automatically save a visual HTML dashboard asset for audit and CI/CD logging
    os.makedirs("outputs", exist_ok=True)
    drift_report_engine.save_html("outputs/data_drift_dashboard.html")
    print("Evidently Data Drift analysis completed. HTML visual log saved to 'outputs/data_drift_dashboard.html'")

    return DriftReport(
        feature_results=results_df,
        drifted_features=tuple(drifted_features_list),
        drift_detected=bool(dataset_drift_detected)
    )

if __name__ == "__main__":
    print("Running integrated Evidently drift monitor pipeline...")
    # Executing the code will generate your structured matrix and output file
    report = monitor_drift()
    print(f"\n===== Production Monitoring Summary =====")
    print(f"Dataset-level Drift Detected: {report.drift_detected}")
    print(f"Number of Drifted Features:   {len(report.drifted_features)}")
    if report.drifted_features:
        print(f"Drifted Feature List:         {report.drifted_features}")
    print(f"=========================================")

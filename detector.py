"""
detector.py — LTE Network KPI Anomaly Detection

Uses Isolation Forest for unsupervised multivariate anomaly detection,
plus threshold-based rules for explainability.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# KPI thresholds based on industry standards
THRESHOLDS = {
    "prb_util_pct":     {"warn": 70,   "crit": 85,   "direction": "high"},
    "drop_rate_pct":    {"warn": 2.0,  "crit": 3.0,  "direction": "high"},
    "rsrp_dbm":         {"warn": -100, "crit": -110, "direction": "low"},
    "sinr_db":          {"warn": 3,    "crit": 0,    "direction": "low"},
    "hosr_pct":         {"warn": 96,   "crit": 93,   "direction": "low"},
    "throughput_mbps":  {"warn": 15,   "crit": 5,    "direction": "low"},
}

KPI_FEATURES = list(THRESHOLDS.keys())


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


def apply_threshold_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows that breach KPI thresholds and explain why."""
    violations = []
    severity = []

    for _, row in df.iterrows():
        row_violations = []
        max_sev = "OK"

        for kpi, thresh in THRESHOLDS.items():
            val = row[kpi]
            if thresh["direction"] == "high":
                if val >= thresh["crit"]:
                    row_violations.append(f"{kpi}={val} (CRITICAL >{thresh['crit']})")
                    max_sev = "CRITICAL"
                elif val >= thresh["warn"]:
                    row_violations.append(f"{kpi}={val} (WARNING >{thresh['warn']})")
                    if max_sev != "CRITICAL":
                        max_sev = "WARNING"
            else:  # low direction
                if val <= thresh["crit"]:
                    row_violations.append(f"{kpi}={val} (CRITICAL <{thresh['crit']})")
                    max_sev = "CRITICAL"
                elif val <= thresh["warn"]:
                    row_violations.append(f"{kpi}={val} (WARNING <{thresh['warn']})")
                    if max_sev != "CRITICAL":
                        max_sev = "WARNING"

        violations.append("; ".join(row_violations) if row_violations else "")
        severity.append(max_sev)

    df = df.copy()
    df["threshold_violations"] = violations
    df["severity"] = severity
    return df


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    """Run Isolation Forest on KPI features and add anomaly scores."""
    X = df[KPI_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    df = df.copy()
    df["anomaly_score"] = model.fit_predict(X_scaled)       # -1 = anomaly, 1 = normal
    df["anomaly_confidence"] = -model.score_samples(X_scaled)  # higher = more anomalous
    df["ml_anomaly"] = df["anomaly_score"] == -1
    return df


def classify_anomaly(row: pd.Series) -> str:
    """Classify the likely type of anomaly based on which KPIs are out of range."""
    if not row["ml_anomaly"] and row["severity"] == "OK":
        return "Normal"

    prb  = row["prb_util_pct"]
    drop = row["drop_rate_pct"]
    rsrp = row["rsrp_dbm"]
    sinr = row["sinr_db"]
    hosr = row["hosr_pct"]
    tput = row["throughput_mbps"]

    if prb > 80 and tput < 20:
        return "Congestion"
    if rsrp < -105 and sinr < 3:
        return "Poor Coverage"
    if sinr < 0 and drop > 2:
        return "Interference"
    if hosr < 94 and drop > 2:
        return "Handover Failures"
    if sum([prb > 75, rsrp < -100, sinr < 3, drop > 2.5]) >= 2:
        return "Multiple Issues"
    return "Anomaly"


def analyze(filepath: str) -> pd.DataFrame:
    """Full pipeline: load → threshold check → ML detection → classify."""
    df = load_data(filepath)
    df = apply_threshold_rules(df)
    df = run_isolation_forest(df)
    df["detected_issue"] = df.apply(classify_anomaly, axis=1)
    df["flagged"] = df["ml_anomaly"] | (df["severity"] != "OK")
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Return summary stats for the dashboard."""
    flagged = df[df["flagged"]]
    return {
        "total_readings":   len(df),
        "total_cells":      df["cell_id"].nunique(),
        "flagged_readings": len(flagged),
        "flagged_cells":    flagged["cell_id"].nunique(),
        "critical_count":   len(df[df["severity"] == "CRITICAL"]),
        "warning_count":    len(df[df["severity"] == "WARNING"]),
        "issue_breakdown":  flagged["detected_issue"].value_counts().to_dict(),
        "worst_cells":      (
            flagged.groupby("cell_id")["anomaly_confidence"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        ),
    }


if __name__ == "__main__":
    df = analyze("data/kpi_data.csv")
    stats = summarize(df)
    print(f"\nTotal readings : {stats['total_readings']}")
    print(f"Flagged        : {stats['flagged_readings']} across {stats['flagged_cells']} cells")
    print(f"Critical       : {stats['critical_count']}")
    print(f"Warning        : {stats['warning_count']}")
    print(f"\nIssue breakdown:")
    for k, v in stats["issue_breakdown"].items():
        print(f"  {k}: {v}")
    print(f"\nWorst cells:")
    for cell, score in stats["worst_cells"].items():
        print(f"  {cell}: {score:.3f}")

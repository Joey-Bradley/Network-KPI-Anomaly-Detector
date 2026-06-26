"""
generate_kpi_data.py — Generate synthetic LTE network KPI data for demo/testing

Run:  python generate_kpi_data.py
Output: data/kpi_data.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_SITES = 50
NUM_HOURS = 24  # one day of hourly data
ANOMALY_RATE = 0.08  # ~8% of readings are anomalous

sites = [f"CELL_{str(i).zfill(3)}" for i in range(1, NUM_SITES + 1)]
hours = pd.date_range("2026-06-01", periods=NUM_HOURS, freq="h")

rows = []
for site in sites:
    for ts in hours:
        # Normal KPI ranges based on real LTE thresholds
        prb_util     = np.random.uniform(20, 65)     # % PRB utilization
        drop_rate    = np.random.uniform(0.2, 1.8)   # % drop call rate
        rsrp         = np.random.uniform(-95, -75)   # dBm (acceptable to good)
        sinr         = np.random.uniform(5, 22)      # dB (acceptable to excellent)
        hosr         = np.random.uniform(97, 99.5)   # % handover success rate
        throughput   = np.random.uniform(30, 120)    # Mbps

        # Inject anomalies randomly
        is_anomaly = np.random.random() < ANOMALY_RATE
        anomaly_type = None

        if is_anomaly:
            scenario = np.random.choice([
                "congestion", "coverage", "interference", "handover", "mixed"
            ])
            if scenario == "congestion":
                prb_util   = np.random.uniform(80, 99)
                throughput = np.random.uniform(2, 15)
                anomaly_type = "High PRB Utilization / Congestion"
            elif scenario == "coverage":
                rsrp       = np.random.uniform(-120, -105)
                sinr       = np.random.uniform(-5, 2)
                drop_rate  = np.random.uniform(3, 8)
                anomaly_type = "Poor Coverage"
            elif scenario == "interference":
                sinr       = np.random.uniform(-8, 0)
                drop_rate  = np.random.uniform(2.5, 6)
                anomaly_type = "Interference"
            elif scenario == "handover":
                hosr       = np.random.uniform(85, 93)
                drop_rate  = np.random.uniform(2, 5)
                anomaly_type = "Handover Failures"
            else:
                prb_util   = np.random.uniform(75, 95)
                rsrp       = np.random.uniform(-115, -100)
                sinr       = np.random.uniform(-3, 3)
                drop_rate  = np.random.uniform(3, 7)
                hosr       = np.random.uniform(88, 94)
                anomaly_type = "Multiple Issues"

        rows.append({
            "timestamp":    ts,
            "cell_id":      site,
            "prb_util_pct": round(prb_util, 2),
            "drop_rate_pct":round(drop_rate, 3),
            "rsrp_dbm":     round(rsrp, 1),
            "sinr_db":      round(sinr, 1),
            "hosr_pct":     round(hosr, 2),
            "throughput_mbps": round(throughput, 1),
            "is_anomaly":   is_anomaly,
            "anomaly_type": anomaly_type if is_anomaly else "Normal",
        })

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("data/kpi_data.csv", index=False)

print(f"Generated {len(df)} records across {NUM_SITES} cells.")
print(f"Anomalies injected: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
print("Saved to data/kpi_data.csv")

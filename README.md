# 📶 Network KPI Anomaly Detector

An AI-powered LTE network performance monitoring tool that detects anomalies in cell site KPI data using machine learning and generates root cause analysis with actionable recommendations.

## What It Does

- Ingests LTE KPI data (CSV) from any source — OSS exports, drive test tools, or synthetic data
- Detects anomalies using **Isolation Forest** (unsupervised ML) + **threshold-based rules**
- Classifies issue types: Congestion, Poor Coverage, Interference, Handover Failures
- Visualizes KPI trends per cell with flagged anomalies highlighted
- Generates **AI root cause analysis** via DeepSeek LLM with specific action recommendations
- Exports flagged records for follow-up

## Tech Stack

| Component | Technology |
|---|---|
| Anomaly Detection | Scikit-learn Isolation Forest |
| Threshold Rules | Custom KPI thresholds (industry standard) |
| Visualization | Plotly + Streamlit |
| AI Analysis | DeepSeek API (deepseek-chat) |
| Data | Pandas / NumPy |

## KPI Thresholds

| KPI | Warning | Critical |
|---|---|---|
| PRB Utilization | >70% | >85% |
| Drop Call Rate | >2% | >3% |
| RSRP | <-100 dBm | <-110 dBm |
| SINR | <3 dB | <0 dB |
| Handover Success Rate | <96% | <93% |
| Throughput | <15 Mbps | <5 Mbps |

## Setup

### 1. Install dependencies
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Generate sample data
```bash
python generate_kpi_data.py
```
Generates 1,200 KPI readings across 50 cell sites with realistic anomaly injection.

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Enter your DeepSeek API key in the sidebar for AI analysis

## Using Your Own Data

Export KPI data from your OSS (Ericsson OSS, Nokia NetAct) as CSV with these columns:

```
timestamp, cell_id, prb_util_pct, drop_rate_pct, rsrp_dbm, sinr_db, hosr_pct, throughput_mbps
```

Upload via the file uploader in the app.

## Project Structure

```
kpi_anomaly_detector/
├── app.py                  # Streamlit web app
├── detector.py             # ML detection engine
├── generate_kpi_data.py    # Synthetic data generator
├── requirements.txt
├── README.md
└── data/
    └── kpi_data.csv        # Auto-created by generate_kpi_data.py
```

## Author

Joey Bradley | [LinkedIn](https://www.linkedin.com/in/joey-bradley-740a2925/) | [GitHub](https://github.com/Joey-Bradley)

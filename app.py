"""
Network KPI Anomaly Detector — Streamlit App
Detects anomalies in LTE network KPI data using Isolation Forest + threshold rules.
LLM (DeepSeek) generates root cause analysis and recommendations.

Usage:
  python generate_kpi_data.py   # generate sample data
  streamlit run app.py
"""

import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
from detector import summarize, KPI_FEATURES, apply_threshold_rules, run_isolation_forest, classify_anomaly

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

st.set_page_config(
    page_title="Network KPI Anomaly Detector",
    page_icon="📶",
    layout="wide"
)

st.title("📶 Network KPI Anomaly Detector")
st.markdown("Upload LTE KPI data to detect anomalies, classify issues, and get AI-powered recommendations.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    server_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key = server_api_key if server_api_key else st.text_input(
        "DeepSeek API Key",
        value="",
        type="password",
        placeholder="sk-...",
        help="Required for AI root cause analysis"
    )
    if server_api_key:
        st.caption("✅ API key configured")
    st.divider()
    st.markdown("**Expected CSV columns:**")
    st.code(
        "timestamp, cell_id\n"
        "prb_util_pct, drop_rate_pct\n"
        "rsrp_dbm, sinr_db\n"
        "hosr_pct, throughput_mbps",
        language="text"
    )
    st.divider()
    st.markdown("**KPI Thresholds:**")
    st.markdown("- PRB Util: Warn >70%, Crit >85%")
    st.markdown("- Drop Rate: Warn >2%, Crit >3%")
    st.markdown("- RSRP: Warn <-100 dBm, Crit <-110 dBm")
    st.markdown("- SINR: Warn <3 dB, Crit <0 dB")
    st.markdown("- HOSR: Warn <96%, Crit <93%")
    st.markdown("- Throughput: Warn <15 Mbps, Crit <5 Mbps")


# ── Data Loading ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload KPI CSV file", type=["csv"])
with col2:
    use_sample = st.button("📊 Load Sample Data", use_container_width=True,
                           help="Uses data/kpi_data.csv — run generate_kpi_data.py first")

df_raw = None
if uploaded:
    df_raw = pd.read_csv(uploaded, parse_dates=["timestamp"])
    st.success(f"Loaded {len(df_raw):,} records from uploaded file.")
elif use_sample:
    if os.path.exists("data/kpi_data.csv"):
        df_raw = pd.read_csv("data/kpi_data.csv", parse_dates=["timestamp"])
        st.success(f"Loaded {len(df_raw):,} records from sample data.")
    else:
        st.error("Sample data not found. Run `python generate_kpi_data.py` first.")

if df_raw is None:
    st.info("Upload a CSV file or load the sample data to get started.")
    st.stop()


# ── Run Detection ─────────────────────────────────────────────────────────────
with st.spinner("Running anomaly detection..."):
    df = apply_threshold_rules(df_raw)
    df = run_isolation_forest(df)
    df["detected_issue"] = df.apply(classify_anomaly, axis=1)
    df["flagged"] = df["ml_anomaly"] | (df["severity"] != "OK")

stats = summarize(df)
flagged_df = df[df["flagged"]].copy()


# ── Summary Cards ─────────────────────────────────────────────────────────────
st.subheader("📊 Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Readings", f"{stats['total_readings']:,}")
c2.metric("Cells Monitored", stats["total_cells"])
c3.metric("🔴 Critical", stats["critical_count"])
c4.metric("🟡 Warning", stats["warning_count"])
c5.metric("Cells Flagged", stats["flagged_cells"])

st.divider()


# ── Issue Breakdown Chart ──────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Issue Breakdown")
    if stats["issue_breakdown"]:
        issue_df = pd.DataFrame(
            list(stats["issue_breakdown"].items()),
            columns=["Issue Type", "Count"]
        ).sort_values("Count", ascending=True)
        fig = px.bar(issue_df, x="Count", y="Issue Type", orientation="h",
                     color="Count", color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Top 10 Worst Cells")
    if stats["worst_cells"]:
        worst_df = pd.DataFrame(
            list(stats["worst_cells"].items()),
            columns=["Cell ID", "Anomaly Score"]
        ).sort_values("Anomaly Score", ascending=True)
        fig2 = px.bar(worst_df, x="Anomaly Score", y="Cell ID", orientation="h",
                      color="Anomaly Score", color_continuous_scale="Oranges")
        fig2.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ── KPI Trend Charts ──────────────────────────────────────────────────────────
st.subheader("📈 KPI Trends")
selected_cell = st.selectbox("Select cell to inspect:", sorted(df["cell_id"].unique()))
cell_df = df[df["cell_id"] == selected_cell].sort_values("timestamp")

kpi_display = {
    "prb_util_pct": "PRB Utilization (%)",
    "drop_rate_pct": "Drop Call Rate (%)",
    "rsrp_dbm": "RSRP (dBm)",
    "sinr_db": "SINR (dB)",
    "hosr_pct": "Handover Success Rate (%)",
    "throughput_mbps": "Throughput (Mbps)"
}

cols = st.columns(2)
for i, (kpi, label) in enumerate(kpi_display.items()):
    with cols[i % 2]:
        fig = go.Figure()
        colors = ["red" if f else "steelblue" for f in cell_df["flagged"]]
        fig.add_trace(go.Scatter(
            x=cell_df["timestamp"], y=cell_df[kpi],
            mode="lines+markers",
            marker=dict(color=colors, size=6),
            line=dict(color="steelblue", width=1.5),
            name=label
        ))
        fig.update_layout(title=label, height=220, margin=dict(t=30, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

st.divider()


# ── Flagged Records Table ──────────────────────────────────────────────────────
st.subheader("🚨 Flagged Records")
severity_filter = st.multiselect(
    "Filter by severity:",
    options=["CRITICAL", "WARNING", "OK"],
    default=["CRITICAL", "WARNING"]
)
filtered = flagged_df[flagged_df["severity"].isin(severity_filter)] if severity_filter else flagged_df

display_cols = ["timestamp", "cell_id", "severity", "detected_issue",
                "prb_util_pct", "drop_rate_pct", "rsrp_dbm", "sinr_db",
                "hosr_pct", "throughput_mbps", "threshold_violations"]
st.dataframe(
    filtered[display_cols].sort_values(["severity", "cell_id"]),
    use_container_width=True,
    height=300
)

csv_buf = io.StringIO()
filtered[display_cols].to_csv(csv_buf, index=False)
st.download_button("⬇️ Export Flagged Records", csv_buf.getvalue(),
                   "flagged_kpis.csv", "text/csv")

st.divider()


# ── AI Root Cause Analysis ────────────────────────────────────────────────────
st.subheader("🤖 AI Root Cause Analysis")
st.markdown("Select a flagged cell and get an AI-generated root cause analysis and action plan.")

flagged_cells = sorted(flagged_df["cell_id"].unique())
if not flagged_cells:
    st.info("No flagged cells to analyze.")
else:
    selected_flagged = st.selectbox("Select a flagged cell:", flagged_cells, key="ai_cell")
    cell_data = flagged_df[flagged_df["cell_id"] == selected_flagged].sort_values("timestamp")

    if st.button("🔍 Analyze with AI", type="primary"):
        if not api_key:
            st.error("Please enter your DeepSeek API key in the sidebar.")
        else:
            avg_kpis = cell_data[KPI_FEATURES].mean().round(2).to_dict()
            issues = cell_data["detected_issue"].value_counts().to_dict()
            violations = "; ".join(cell_data["threshold_violations"].dropna().unique()[:5])

            prompt = f"""You are a senior LTE network performance engineer.
Analyze the following KPI data for cell {selected_flagged} and provide:
1. Root cause assessment
2. Likely contributing factors
3. Recommended actions (specific, prioritized)
4. Any additional monitoring needed

Cell ID: {selected_flagged}
Average KPIs (last period):
- PRB Utilization: {avg_kpis['prb_util_pct']}%
- Drop Call Rate: {avg_kpis['drop_rate_pct']}%
- RSRP: {avg_kpis['rsrp_dbm']} dBm
- SINR: {avg_kpis['sinr_db']} dB
- Handover Success Rate: {avg_kpis['hosr_pct']}%
- Throughput: {avg_kpis['throughput_mbps']} Mbps

Detected issues: {issues}
Threshold violations: {violations}

Provide a concise, technical analysis with specific action items."""

            with st.spinner("Generating root cause analysis..."):
                try:
                    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
                    response = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=800
                    )
                    analysis = response.choices[0].message.content
                    st.session_state["ai_analysis"] = analysis
                    st.session_state["ai_analysis_cell"] = selected_flagged
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")

    if "ai_analysis" in st.session_state and st.session_state.get("ai_analysis_cell") == selected_flagged:
        st.markdown(f"### Analysis for {selected_flagged}")
        st.markdown(st.session_state["ai_analysis"])

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Network KPI Anomaly Detector | Isolation Forest + DeepSeek AI | github.com/Joey-Bradley")

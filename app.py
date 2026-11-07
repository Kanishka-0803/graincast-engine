import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
st.set_page_config(
    page_title="Paddy Procurement Copilot | TNCSC Intelligence",
    page_icon="*",
    layout="wide"
)
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
MODELS_DIR = Path("models")
MARKET_OPTIONS = {
    "Cauvery Delta Pooled": {
        "signals": "signals_cauvery_delta_pooled.csv",
        "plot": "forecast_eval_cauvery_delta_pooled.png",
        "importance": "importance_cauvery_delta_pooled.csv",
        "desc": "Regional median aggregation across Thanjavur, Tiruvarur, Nagapattinam & Mayiladuthurai."
    },
    "Madurai": {
        "signals": "signals_madurai.csv",
        "plot": "forecast_eval_madurai.png",
        "importance": "importance_madurai.csv",
        "desc": "South-Central Tamil Nadu benchmark regulated market."
    },
    "Cuddalore": {
        "signals": "signals_cuddalore.csv",
        "plot": "forecast_eval_cuddalore.png",
        "importance": "importance_cuddalore.csv",
        "desc": "Coastal Northern Tamil Nadu market series."
    }
}
st.title("GrainCast: Paddy Procurement Copilot")
st.caption("AI-powered 1-week forward price forecasting and MSP arbitrage engine for Tamil Nadu Civil Supplies Corporation (TNCSC).")
st.sidebar.header("Mandi & Region Selection")
selected_market = st.sidebar.selectbox("Choose Mandi / Pool", list(MARKET_OPTIONS.keys()))
market_meta = MARKET_OPTIONS[selected_market]
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Region Profile:**\n{market_meta['desc']}")
st.sidebar.markdown("**Statutory Baseline (2024-2025):**\nMSP Floor: `Rs 2,300.0 / Quintal`")
signal_path = PROCESSED_DIR / market_meta["signals"]
if not signal_path.exists():
    st.error(f"Signal file {signal_path} not found. Please verify data processing.")
    st.stop()
df_signals = pd.read_csv(signal_path)
df_signals["week_start"] = pd.to_datetime(df_signals["week_start"])
latest = df_signals.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Latest Reported Modal Price",
        value=f"Rs {latest['modal_price_interpolated']:.1f} / q"
    )
with col2:
    price_diff = latest['pred_price_t1'] - latest['modal_price_interpolated']
    st.metric(
        label="LightGBM 1-Wk Forecast",
        value=f"Rs {latest['pred_price_t1']:.1f} / q",
        delta=f"{price_diff:+.1f} Rs"
    )
with col3:
    st.metric(
        label="MSP Floor Benchmark",
        value=f"Rs {latest['msp']:.0f} / q",
        delta=f"Spread: {latest['msp_spread_pct']:+.1f}%",
        delta_color="inverse" if latest['msp_spread_pct'] < -3.0 else "normal"
    )
with col4:
    signal_text = str(latest['procurement_signal'])
    if "BUY_DPC_FLOOR" in signal_text:
        badge_color = "red"
        action = "OPEN DPC CENTRES"
    elif "OPEN_MARKET" in signal_text:
        badge_color = "green"
        action = "TRADER ABSORPTION"
    else:
        badge_color = "blue"
        action = "EQUILIBRIUM"
    st.markdown("**Procurement Recommendation:**")
    st.markdown(f":{badge_color}[### {action}]")
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Forecast vs Ground Truth", "Historical Signal Stream", "Model Explainability"])
with tab1:
    st.subheader(f"Out-of-Sample Temporal Validation (2024-2025): {selected_market}")
    plot_file = FIGURES_DIR / market_meta["plot"]
    if plot_file.exists():
        st.image(Image.open(plot_file), use_container_width=True)
    else:
        st.warning("Evaluation plot not found.")
with tab2:
    st.subheader("Recent Weekly Procurement Log")
    display_df = df_signals.tail(15)[["week_start", "modal_price_interpolated", "pred_price_t1", "msp", "msp_spread_pct", "procurement_signal"]].copy()
    display_df["week_start"] = display_df["week_start"].dt.strftime("%Y-%m-%d")
    display_df = display_df.rename(columns={
        "week_start": "Week Date",
        "modal_price_interpolated": "Reported Modal (Rs/q)",
        "pred_price_t1": "Forecast Price (Rs/q)",
        "msp": "MSP (Rs/q)",
        "msp_spread_pct": "Spread vs MSP (%)",
        "procurement_signal": "Signal"
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
with tab3:
    st.subheader("LightGBM Feature Importance Rankings")
    imp_file = MODELS_DIR / market_meta["importance"]
    if imp_file.exists():
        df_imp = pd.read_csv(imp_file)
        st.bar_chart(df_imp.set_index("feature")["importance"])
        st.dataframe(df_imp, use_container_width=True, hide_index=True)
    else:
        st.warning("Feature importance file not found.")

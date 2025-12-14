# main.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="MarketSense", layout="wide")

# =========================
# Config
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"

HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

# =========================
# Load model
# =========================
@st.cache_resource
def load_model_cached(path):
    return tf.keras.models.load_model(path, compile=False)

model = load_model_cached(MODEL_PATH)

# =========================
# Safe yfinance fetch (MATCHES YOUR WORKING LOGIC)
# =========================
@st.cache_data(ttl=600)
def fetch_yf(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df.empty:
        return df

    df = df.reset_index()

    # FIX multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    return df

# =========================
# ATR
# =========================
def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

# =========================
# Build model input (SAME LOGIC AS TRAINING)
# =========================
def build_features(df):
    timesteps = model.input_shape[1]
    close = df["Close"].values

    if len(close) < timesteps:
        raise ValueError("Data tidak cukup untuk window model")

    window = close[-timesteps:]

    # ⚠️ HARUS SAMA DENGAN TRAINING
    window = (window - window.min()) / (window.max() - window.min() + 1e-9)

    return window.reshape(1, timesteps, 1).astype(np.float32)

# =========================
# UI
# =========================
st.title("MarketSense")
st.markdown("AI Forecast + ATR Zone (Yahoo Finance) 📈")

c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.text_input("Ticker Saham", "BBCA.JK")
with c2:
    strategy = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))
with c3:
    horizon_label = st.selectbox("Horizon", ("5 hari kedepan", "10 hari kedepan"))

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# =========================
# Run
# =========================
if st.button("Submit"):
    df = fetch_yf(ticker, start_date, end_date)

    if df.empty:
        st.error("Data kosong. Pastikan ticker benar (IDX pakai .JK)")
        st.stop()

    # ATR
    atr = compute_atr(df)
    atr_last = atr.dropna().iloc[-1]
    atr_mult = ATR_MULT[strategy]

    # Model prediction
    X = build_features(df)
    y_pred = model.predict(X, verbose=0).squeeze()

    horizon = HORIZON_MAP[horizon_label]
    last_close = df["Close"].iloc[-1]

    ai_trend = np.linspace(last_close, y_pred, horizon)

    future_dates = pd.bdate_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)

    upper = ai_trend + atr_last * atr_mult
    lower = ai_trend - atr_last * atr_mult

    # =========================
    # Plot
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=ai_trend,
        mode="lines",
        name="AI Trend",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        fill="tonexty",
        name="ATR Zone",
        opacity=0.3
    ))

    fig.update_layout(
        title=f"{ticker} – AI Forecast + ATR",
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf

# CRITICAL: Disable curl_cffi to prevent impersonation errors
import sys
sys.modules['curl_cffi'] = None

st.set_page_config(page_title="MarketSense", layout="wide")

# =========================
# Config
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"   # put .h5 next to main.py

HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error("Model file tidak ditemukan. Pastikan `LSTM_CNN_model.h5` ada satu folder dengan `main.py`.")
    st.stop()

try:
    model = load_model_cached(str(MODEL_PATH))
except Exception as e:
    st.error("Gagal load model .h5")
    st.exception(e)
    st.stop()

# =========================
# Robust yfinance fetch (retry + candidates) - FIXED FOR STREAMLIT CLOUD
# =========================
@st.cache_data(ttl=600)
def fetch_yf_ohlc(ticker: str, start, end) -> pd.DataFrame:
    def _download(sym: str) -> pd.DataFrame:
        try:
            # Force use of requests library instead of curl_cffi
            import requests
            session = requests.Session()
            session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            
            ticker_obj = yf.Ticker(sym, session=session)
            df = ticker_obj.history(start=start, end=end, auto_adjust=False)
            
        except Exception as e:
            return pd.DataFrame()
        
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.reset_index()

        # flatten multiindex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # normalize date col
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        needed = {"Date", "Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
        return df

    t = (ticker or "").strip().upper()

    # try both with/without .JK automatically
    candidates = [t]
    if t.endswith(".JK"):
        candidates.append(t.replace(".JK", ""))
    else:
        candidates.append(t + ".JK")

    last_err = None
    for sym in candidates:
        for attempt in range(3):
            try:
                df = _download(sym)
                if not df.empty:
                    return df
                # if empty, wait and retry
                time.sleep(1.0 + attempt * 1.0)
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt * 1.0)

    # still empty => give informative message
    msg = (
        f"Yahoo Finance returned empty for {ticker} (tried: {candidates}). "
        "Kemungkinan: (1) Ticker tidak valid, (2) Market sedang tutup, atau (3) Yahoo Finance sedang down."
    )
    if last_err is not None:
        msg += f" Error: {str(last_err)[:200]}"

    return pd.DataFrame({"__error__": [msg]})

# =========================
# ATR
# =========================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()

# =========================
# Build input for model (Close-only template)
# =========================
def build_features_for_model(df: pd.DataFrame) -> np.ndarray:
    inp = model.input_shape  # e.g. (None, 100, 1)

    if isinstance(inp, list):
        raise ValueError(f"Model multi-input terdeteksi: {inp}. Builder perlu disesuaikan.")

    if len(inp) == 3:
        _, timesteps, n_features = inp
        channels = None
    elif len(inp) == 4:
        _, timesteps, n_features, channels = inp
    else:
        raise ValueError(f"Unexpected input shape: {inp}")

    close = df["Close"].astype(float).to_numpy()
    if len(close) < timesteps:
        raise ValueError(f"Data kurang. Butuh minimal {timesteps} bar, tapi hanya ada {len(close)}.")

    window = close[-timesteps:]

    # IMPORTANT: scaling must match training
    # Here we do a local min-max on the window (works only if training did similar)
    wmin, wmax = window.min(), window.max()
    window = (window - wmin) / (wmax - wmin + 1e-9)

    if n_features != 1:
        raise ValueError(
            f"Model butuh n_features={n_features}, tapi builder ini hanya menyiapkan 1 fitur (Close). "
            "Kalau training kamu pakai OHLCV/indikator, ubah builder untuk output (timesteps, n_features)."
        )

    X = window.reshape(1, timesteps, 1).astype(np.float32)
    if channels == 1:
        X = X[..., np.newaxis]  # (1, timesteps, 1, 1)

    return X

# =========================
# Forecast helpers + plotting like your screenshot
# =========================
def make_ai_trend(y_pred: np.ndarray, last_close: float, horizon: int) -> np.ndarray:
    y = np.array(y_pred).squeeze()

    if y.ndim == 0:
        end_val = float(y)
        return np.linspace(last_close, end_val, num=horizon + 1)[1:]

    if y.ndim == 1:
        if len(y) == horizon:
            return y.astype(float)
        # resample to horizon
        x_old = np.linspace(0, 1, num=len(y))
        x_new = np.linspace(0, 1, num=horizon)
        return np.interp(x_new, x_old, y).astype(float)

    raise ValueError(f"Output shape tidak didukung: {np.array(y_pred).shape}")

def plot_forecast_like_screenshot(
    df: pd.DataFrame,
    ticker: str,
    ai_trend: np.ndarray,
    atr_last: float,
    atr_mult: float,
    horizon: int,
):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    ai_series = pd.Series(ai_trend, index=future_dates)

    upper = ai_series + (atr_last * atr_mult if atr_last > 0 else 0.0)
    lower = ai_series - (atr_last * atr_mult if atr_last > 0 else 0.0)

    fig = go.Figure()

    # Harga historis (line + markers)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines+markers",
            name="Harga Historis",
        )
    )

    # Zona aman (fill between upper and lower)
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower.values,
            mode="lines",
            fill="tonexty",
            name=f"Zona Aman (AI + ATR {atr_mult:.1f}x)",
            opacity=0.25,
        )
    )

    # AI trend dashed
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=ai_series.values,
            mode="lines",
            name="AI Trend (Daily Adjusted)",
            line=dict(dash="dash"),
        )
    )

    # Resistance / Support dotted
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper.values,
            mode="lines",
            name="Resistance",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower.values,
            mode="lines",
            name="Support",
            line=dict(dash="dot"),
        )
    )

    fig.update_layout(
        title=f"Final Forecast {ticker.upper()}: AI Horizon Adjusted ( / {horizon})",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        height=580,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(x=0.01, y=0.99),
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# UI (simple, clean)
# =========================
st.title("MarketSense")
st.markdown("AI Forecast + ATR Zone (Yahoo Finance) 📈")

c1, c2, c3 = st.columns([2.2, 1.2, 1.2])
with c1:
    nama_saham = st.text_input("Ticker Saham", placeholder="contoh: BBCA.JK / BBNI.JK")
with c2:
    strategi = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))
with c3:
    horizon_label = st.selectbox("Horizon", ("5 hari kedepan", "10 hari kedepan"))

d1, d2 = st.columns(2)

# Smart defaults: 1 year of data
from datetime import datetime, timedelta
default_end = datetime.now().date()
default_start = default_end - timedelta(days=365)

with d1:
    start_date = st.date_input("Start Date", value=default_start)
with d2:
    end_date = st.date_input("End Date", value=default_end)

if st.button("Submit"):
    ticker = (nama_saham or "").strip()
    if not ticker:
        st.error("Ticker saham belum diisi.")
        st.stop()

    if start_date >= end_date:
        st.error("Start Date harus lebih kecil dari End Date.")
        st.stop()

    horizon = HORIZON_MAP[horizon_label]
    atr_mult = ATR_MULT[strategi]

    with st.spinner("Fetching data from Yahoo Finance..."):
        df = fetch_yf_ohlc(ticker, start_date, end_date)

    # show yf error message clearly
    if "__error__" in df.columns:
        st.error(df["__error__"].iloc[0])
        st.stop()

    if df.empty:
        st.error("Data kosong dari Yahoo Finance.")
        st.stop()

    # ATR
    atr = compute_atr(df, period=14)
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0

    try:
        with st.spinner("Running AI prediction..."):
            X = build_features_for_model(df)
            y_pred = model.predict(X, verbose=0)

            last_close = float(df["Close"].iloc[-1])
            ai_trend = make_ai_trend(y_pred, last_close=last_close, horizon=horizon)

        plot_forecast_like_screenshot(
            df=df,
            ticker=ticker,
            ai_trend=ai_trend,
            atr_last=atr_last,
            atr_mult=atr_mult,
            horizon=horizon,
        )

    except Exception as e:
        st.error("Gagal membuat forecast.")
        st.exception(e)
        st.info(
            "Kalau error `n_features mismatch`, berarti model training pakai fitur > 1 (OHLCV/indikator). "
            "Ubah `build_features_for_model()` agar sesuai `model.input_shape` dan scaling training."
        )
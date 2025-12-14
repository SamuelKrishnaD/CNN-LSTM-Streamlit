# main.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import requests
from pathlib import Path

st.set_page_config(page_title="MarketSense", layout="wide")

# =========================
# CONFIG
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"

TWELVE_DATA_API_KEY = "0fa9874ef304455b944f00d57ed3473e"

HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

DAYS_BACK = 1825  # ✅ 5 YEARS (calendar days)

INDONESIAN_TICKERS = [
    "BBCA","BBRI","BBNI","BMRI","TLKM","ASII","UNVR","ICBP","INDF","KLBF",
    "GGRM","HMSP","SMGR","JSMR","PTBA","ADRO","ITMG","ANTM","INCO","EXCL","GOTO"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error("Model `LSTM_CNN_model.h5` tidak ditemukan.")
    st.stop()

model = load_model_cached(str(MODEL_PATH))

# =========================
# FX RATE USD → IDR
# =========================
@st.cache_data(ttl=3600)
def fetch_usd_idr_rate() -> float:
    url = "https://api.twelvedata.com/exchange_rate"
    params = {"symbol": "USD/IDR", "apikey": TWELVE_DATA_API_KEY}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "rate" not in data:
        raise ValueError("Gagal fetch USD/IDR rate")
    return float(data["rate"])

# =========================
# CURRENCY DETECTOR
# =========================
def detect_currency(ticker: str):
    t = ticker.upper().replace(".JK","")
    if t in INDONESIAN_TICKERS:
        return "Rp", "IDR"
    return "$", "USD"

# =========================
# FETCH DATA (5 YEARS)
# =========================
@st.cache_data(ttl=900)
def fetch_twelve_data(ticker: str) -> pd.DataFrame:
    symbol = ticker.upper().replace(".JK","")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": min(DAYS_BACK + 50, 5000),
        "order": "ASC",
        "format": "JSON"
    }

    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if data.get("status") == "error":
        return pd.DataFrame({"__error__":[data.get("message","API error")]})

    if "values" not in data:
        return pd.DataFrame({"__error__":["No data returned"]})

    df = pd.DataFrame(data["values"])
    df["Date"] = pd.to_datetime(df["datetime"])
    df["Open"] = pd.to_numeric(df["open"])
    df["High"] = pd.to_numeric(df["high"])
    df["Low"]  = pd.to_numeric(df["low"])
    df["Close"]= pd.to_numeric(df["close"])
    df["Volume"]= pd.to_numeric(df["volume"], errors="coerce")

    df = df[["Date","Open","High","Low","Close","Volume"]]
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    # ✅ USD → IDR conversion for IDX
    if symbol in INDONESIAN_TICKERS:
        usd_idr = fetch_usd_idr_rate()
        df[["Open","High","Low","Close"]] *= usd_idr

    return df

# =========================
# ATR
# =========================
def compute_atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =========================
# BUILD FEATURES (MATCH TRAINING)
# =========================
def build_features_for_model(df):
    _, timesteps, n_features = model.input_shape

    d = df.copy()
    d["LogReturn"] = np.log(d["Close"]/d["Close"].shift(1))
    for i in range(1,4):
        d[f"LogReturn_Lag{i}"] = d["LogReturn"].shift(i)

    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        (d["High"] - d["Low"]).abs(),
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - 100/(1+rs)

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["VolChange"] = d["Volume"].pct_change()

    features = [
        "Close","Volume","LogReturn","LogReturn_Lag1","LogReturn_Lag2",
        "LogReturn_Lag3","ATR_14","RSI_14","MACD","VolChange"
    ]

    d = d[features].dropna()

    if len(d) < timesteps:
        raise ValueError("Data kurang untuk window model")

    window = d.tail(timesteps).values
    window = (window - window.min(0)) / (window.max(0) - window.min(0) + 1e-9)

    return window.reshape(1, timesteps, n_features).astype(np.float32)

# =========================
# AI TREND
# =========================
def make_ai_trend(y_pred, last_close, horizon):
    y = np.squeeze(y_pred)
    if y.ndim == 1 and len(y)==2:
        avg_ret = y.mean()
    else:
        avg_ret = float(y)
    end_price = last_close * (1 + avg_ret)
    return np.linspace(last_close, end_price, horizon+1)[1:]

# =========================
# UI
# =========================
st.title("📊 MarketSense")
st.caption("AI Forecast + ATR Zone | **5 Years Historical Data**")

c1,c2,c3 = st.columns(3)
with c1:
    ticker = st.text_input("Ticker", placeholder="BBCA / AAPL")
with c2:
    strategy = st.selectbox("Strategy", ATR_MULT.keys())
with c3:
    horizon_label = st.selectbox("Horizon", HORIZON_MAP.keys())

if st.button("🔮 Analyze & Predict", use_container_width=True):
    symbol = ticker.strip().upper()
    if not symbol:
        st.stop()

    df = fetch_twelve_data(symbol)
    if "__error__" in df.columns:
        st.error(df["__error__"][0])
        st.stop()

    atr = compute_atr(df).iloc[-1]
    X = build_features_for_model(df)
    y_pred = model.predict(X, verbose=0)

    last_close = df["Close"].iloc[-1]
    horizon = HORIZON_MAP[horizon_label]
    ai_trend = make_ai_trend(y_pred, last_close, horizon)

    currency_symbol, _ = detect_currency(symbol)

    st.success(f"Using **{len(df)} bars (~5 years)**")

    st.metric("Last Close", f"{currency_symbol}{last_close:,.0f}")
    st.metric("Predicted", f"{currency_symbol}{ai_trend[-1]:,.0f}")

    st.line_chart(pd.Series(ai_trend))

st.caption("⚠️ Educational purpose only")

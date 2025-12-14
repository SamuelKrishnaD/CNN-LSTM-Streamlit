import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import requests
from pathlib import Path

st.set_page_config(page_title="MarketSense", layout="wide")

# =========================
# Config
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"

STRATEGY_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

# =========================
# Secrets
# =========================
API_KEY = st.secrets.get("TWELVEDATA_API_KEY", "")
if not API_KEY:
    st.error("TWELVEDATA_API_KEY belum diset. Tambahkan di Streamlit Secrets.")
    st.stop()

# =========================
# Load model
# =========================
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error("Model file tidak ditemukan. Pastikan `LSTM_CNN_model.h5` ada satu folder dengan `main.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))

# =========================
# Twelve Data fetch
# =========================
@st.cache_data(ttl=60 * 10)
def fetch_ohlc_twelvedata(symbol: str, interval: str = "1day", outputsize: int = 260) -> pd.DataFrame:
    """
    Returns OHLCV dataframe with columns: Date, Open, High, Low, Close, Volume
    Twelve Data docs: time_series endpoint
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON",
        "order": "ASC",
    }

    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    # Handle API errors
    if isinstance(data, dict) and data.get("status") == "error":
        msg = data.get("message", "Unknown error from Twelve Data")
        return pd.DataFrame({"__error__": [msg]})

    values = data.get("values")
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)

    # Expected columns in values: datetime, open, high, low, close, volume
    colmap = {
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=colmap)

    needed = {"Date", "Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = np.nan

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
    return df.reset_index(drop=True)

# =========================
# ATR
# =========================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()

# =========================
# Model input builder (TEMPLATE)
# =========================
def build_features_for_model(df: pd.DataFrame) -> np.ndarray:
    inp = model.input_shape

    if isinstance(inp, list):
        raise ValueError(f"Model multi-input: {inp} (butuh builder khusus).")

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

    # TODO: scaling sama seperti training (IMPORTANT)
    # window = (window - window.min()) / (window.max() - window.min() + 1e-9)

    if n_features != 1:
        raise ValueError(
            f"Model butuh n_features={n_features}, tapi builder ini hanya menyiapkan 1 fitur (Close). "
            "Ubah sesuai pipeline training kamu (misal OHLCV/indikator)."
        )

    X = window.reshape(1, timesteps, 1).astype(np.float32)
    if channels == 1:
        X = X[..., np.newaxis]
    return X

# =========================
# AI Trend maker
# =========================
def make_ai_trend(y_pred: np.ndarray, last_close: float, horizon: int) -> np.ndarray:
    y = np.array(y_pred).squeeze()

    if y.ndim == 0:
        end_val = float(y)
        return np.linspace(last_close, end_val, num=horizon + 1)[1:]

    if y.ndim == 1:
        if len(y) == horizon:
            return y.astype(float)
        x_old = np.linspace(0, 1, num=len(y))
        x_new = np.linspace(0, 1, num=horizon)
        return np.interp(x_new, x_old, y).astype(float)

    raise ValueError(f"Output shape tidak didukung: {np.array(y_pred).shape}")

# =========================
# Plot like screenshot
# =========================
def plot_forecast(df: pd.DataFrame, ticker: str, ai_trend: np.ndarray, atr_last: float, atr_mult: float, horizon: int):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    ai_series = pd.Series(ai_trend, index=future_dates)
    upper = ai_series + atr_last * atr_mult
    lower = ai_series - atr_last * atr_mult

    fig = go.Figure()

    # Harga Historis
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines+markers",
        name="Harga Historis"
    ))

    # Zona Aman
    fig.add_trace(go.Scatter(
        x=future_dates, y=upper.values,
        mode="lines", line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lower.values,
        mode="lines",
        fill="tonexty",
        name=f"Zona Aman (AI + ATR {atr_mult:.1f}x)",
        opacity=0.25
    ))

    # AI Trend
    fig.add_trace(go.Scatter(
        x=future_dates, y=ai_series.values,
        mode="lines",
        name="AI Trend (Daily Adjusted)",
        line=dict(dash="dash")
    ))

    # Resistance / Support
    fig.add_trace(go.Scatter(
        x=future_dates, y=upper.values,
        mode="lines",
        name="Resistance",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lower.values,
        mode="lines",
        name="Support",
        line=dict(dash="dot")
    ))

    fig.update_layout(
        title=f"Final Forecast {ticker.upper()}: AI Horizon Adjusted ( / {horizon})",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(x=0.01, y=0.99),
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# UI
# =========================
st.title("MarketSense")
st.markdown("IDX forecast pakai Twelve Data (stabil di Streamlit Cloud). 📈🟩")

c1, c2, c3 = st.columns([2.2, 1.2, 1.2])
with c1:
    nama_saham = st.text_input("Ticker saham (IDX)", placeholder="contoh: BBCA.JK / BBNI.JK")
with c2:
    jenis_strategi = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))
with c3:
    jangka_prediksi = st.selectbox("Jangka waktu", ("5 hari kedepan", "10 hari kedepan"))

if st.button("Submit"):
    ticker = nama_saham.strip()
    if not ticker:
        st.error("Ticker belum diisi.")
        st.stop()

    horizon = HORIZON_MAP[jangka_prediksi]
    atr_mult = ATR_MULT[jenis_strategi]

    df = fetch_ohlc_twelvedata(ticker, interval="1day", outputsize=300)

    # If Twelve Data returns an error message, show it
    if "__error__" in df.columns:
        st.error(f"Twelve Data error: {df['__error__'].iloc[0]}")
        st.stop()

    if df.empty:
        st.error("Data kosong dari Twelve Data. Cek ticker (contoh IDX: BBCA.JK).")
        st.stop()

    atr = compute_atr(df, period=14)
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0

    try:
        X = build_features_for_model(df)
        y_pred = model.predict(X, verbose=0)

        last_close = float(df["Close"].iloc[-1])
        ai_trend = make_ai_trend(y_pred, last_close=last_close, horizon=horizon)

        plot_forecast(
            df=df,
            ticker=ticker,
            ai_trend=ai_trend,
            atr_last=atr_last,
            atr_mult=atr_mult if atr_last > 0 else 0.0,
            horizon=horizon,
        )

    except Exception as e:
        st.error("Gagal membuat forecast.")
        st.exception(e)
        st.info(
            "Kalau error `n_features mismatch`, berarti model training pakai fitur > 1 (OHLCV/indikator). "
            "Ubah `build_features_for_model()` agar sesuai `model.input_shape` + scaler training."
        )

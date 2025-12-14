import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="MarketSense", layout="wide")

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"

STRATEGY_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error("Model file tidak ditemukan. Pastikan `LSTM_CNN_model.h5` ada satu folder dengan `main.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))

# -------------------------
# Robust fetch OHLCV
# -------------------------
@st.cache_data(ttl=60 * 10)
def fetch_ohlc_robust(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Primary: yf.Ticker(ticker).history() (usually more stable)
    Fallback: yf.download()
    """
    # 1) Primary
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, auto_adjust=False)
        if hist is not None and not hist.empty:
            hist = hist.reset_index()
            # rename Datetime -> Date if needed
            if "Datetime" in hist.columns and "Date" not in hist.columns:
                hist = hist.rename(columns={"Datetime": "Date"})
            if "Date" not in hist.columns and "index" in hist.columns:
                hist = hist.rename(columns={"index": "Date"})

            needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
            if needed.issubset(hist.columns):
                hist = hist.dropna(subset=["Date", "Open", "High", "Low", "Close"])
                return hist
    except Exception:
        pass

    # 2) Fallback
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        cols = {c: (c[0] if isinstance(c, tuple) else c) for c in df.columns}
        df = df.rename(columns=cols)

        needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------
# ATR
# -------------------------
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

# -------------------------
# Model input builder (TEMPLATE)
# -------------------------
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

# -------------------------
# AI Trend maker
# -------------------------
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

# -------------------------
# Plot like screenshot
# -------------------------
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

    # Zona Aman (fill between upper and lower)
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

    # AI Trend (dashed)
    fig.add_trace(go.Scatter(
        x=future_dates, y=ai_series.values,
        mode="lines",
        name="AI Trend (Daily Adjusted)",
        line=dict(dash="dash")
    ))

    # Resistance / Support (dotted)
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
st.markdown("Forecast + Zona Aman ATR (Support/Resistance)")

c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2])
with c1:
    nama_saham = st.text_input("Ticker saham", placeholder="contoh: BBCA.JK / BBNI.JK / AAPL")
with c2:
    jenis_strategi = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))
with c3:
    jangka_prediksi = st.selectbox("Jangka waktu", ("5 hari kedepan", "10 hari kedepan"))
with c4:
    period = st.selectbox("Data historis", ("3mo", "6mo", "1y", "2y", "5y"), index=2)

if st.button("Submit"):
    ticker = nama_saham.strip()
    if not ticker:
        st.error("Ticker belum diisi.")
        st.stop()

    horizon = HORIZON_MAP[jangka_prediksi]
    atr_mult = ATR_MULT[jenis_strategi]

    df = fetch_ohlc_robust(ticker, period=period, interval="1d")
    if df.empty:
        st.error(
            "Gagal ambil data dari Yahoo Finance (sering kena rate limit / bot protection di Streamlit Cloud).\n\n"
            "Coba lagi beberapa menit, atau coba ticker lain. Untuk Indonesia biasanya pakai `.JK` (BBCA.JK)."
        )
        st.stop()

    # ATR
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
            "Ubah `build_features_for_model()` sesuai pipeline training + scaler."
        )

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
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"  # keep .h5 in same folder as main.py

STRATEGY_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model(path: str):
    # compile=False helps when .h5 contains training-time custom metrics/loss
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error("Model file tidak ditemukan. Pastikan `LSTM_CNN_model.h5` ada satu folder dengan `main.py`.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error("Gagal load model .h5")
    st.exception(e)
    st.stop()

# =========================
# Fetch OHLCV via yfinance (Streamlit Cloud safe)
# =========================
@st.cache_data(ttl=60 * 10)
def fetch_ohlc(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    # threads=False helps avoid the "Impersonating chromeXXX is not supported" issue in some environments
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

    # Normalize column names (yfinance can return multiindex in some cases)
    # Ensure we have Date, Open, High, Low, Close, Volume
    cols = {c: c for c in df.columns}
    for c in list(cols.keys()):
        if isinstance(c, tuple) and len(c) > 0:
            cols[c] = c[0]
    df = df.rename(columns=cols)

    needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    # drop rows with missing values
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    return df

# =========================
# Chart
# =========================
def plot_stock_chart(df: pd.DataFrame, title: str, pred_series: pd.Series | None = None):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close",
        )
    )

    if pred_series is not None and len(pred_series) > 0:
        fig.add_trace(
            go.Scatter(
                x=pred_series.index,
                y=pred_series.values,
                mode="lines+markers",
                name="Forecast",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# Build input for model (TEMPLATE)
# IMPORTANT: adapt to your training pipeline (features + scaler + window length)
# =========================
def build_features_for_model(df: pd.DataFrame) -> np.ndarray:
    inp = model.input_shape  # e.g. (None, 60, 1) or (None, 60, 5) or (None, 60, 5, 1)

    if isinstance(inp, list):
        raise ValueError(f"Model multi-input terdeteksi: {inp}. Builder perlu disesuaikan.")

    if len(inp) == 3:
        _, timesteps, n_features = inp
        channels = None
    elif len(inp) == 4:
        _, timesteps, n_features, channels = inp
    else:
        raise ValueError(f"Unexpected input shape: {inp}")

    # ---- Default builder: Close-only ----
    close = df["Close"].astype(float).to_numpy()
    if len(close) < timesteps:
        raise ValueError(f"Data kurang. Butuh minimal {timesteps} bar, tapi hanya ada {len(close)}.")

    window = close[-timesteps:]

    # TODO: Apply SAME scaler as training (important!)
    # Example MinMax on window (NOT ideal unless training used same logic):
    # window = (window - window.min()) / (window.max() - window.min() + 1e-9)

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
# UI
# =========================
st.title("MarketSense")
st.markdown("Tampilkan chart historis dari Yahoo Finance dan hasil forecast dari model CNN-LSTM. 📈")

c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2])

with c1:
    nama_saham = st.text_input("Ticker saham (Yahoo Finance)", placeholder="contoh: BBCA.JK / AAPL / TSLA")

with c2:
    jenis_strategi = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))

with c3:
    jangka_prediksi = st.selectbox("Jangka waktu", ("5 hari kedepan", "10 hari kedepan"))

with c4:
    period = st.selectbox("Data historis", ("3mo", "6mo", "1y", "2y", "5y"), index=2)

# =========================
# Action
# =========================
if st.button("Submit"):
    ticker = nama_saham.strip()
    if not ticker:
        st.error("Ticker saham belum diisi.")
        st.stop()

    horizon = HORIZON_MAP[jangka_prediksi]
    strategy_id = STRATEGY_MAP[jenis_strategi]

    # Fetch data
    df = fetch_ohlc(ticker, period=period, interval="1d")
    if df.empty:
        st.error(
            "Data tidak ditemukan / gagal download. "
            "Untuk saham Indonesia biasanya perlu suffix `.JK` (contoh: BBCA.JK)."
        )
        st.stop()

    st.subheader("Chart Historis")
    plot_stock_chart(df, f"{ticker.upper()} | Period: {period}")

    # Predict
    try:
        X = build_features_for_model(df)
        y_pred = model.predict(X, verbose=0)
        y_flat = np.array(y_pred).squeeze()

        # Convert prediction to a 1D series for plotting
        if y_flat.ndim == 0:
            # single value -> repeat horizon
            y_flat = np.repeat(float(y_flat), horizon)
        elif y_flat.ndim == 1:
            # ok (sequence)
            pass
        else:
            raise ValueError(f"Output shape tidak didukung untuk chart: {np.array(y_pred).shape}")

        # future dates (business days)
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=len(y_flat))
        forecast = pd.Series(y_flat.astype(float), index=future_dates)

        # (Optional) Anchor forecast to last close if your model outputs deltas/returns
        # last_close = float(df["Close"].iloc[-1])
        # forecast = last_close + forecast.cumsum()

        st.subheader("Chart dengan Forecast")
        plot_stock_chart(df, f"{ticker.upper()} + Forecast ({jangka_prediksi})", pred_series=forecast)

        st.caption(f"Strategy: {jenis_strategi} (id={strategy_id}) | Horizon: {horizon} hari")

        with st.expander("Nilai forecast"):
            st.dataframe(forecast.to_frame("Forecast"))

    except Exception as e:
        st.error("Gagal melakukan prediksi.")
        st.exception(e)
        st.info(
            "Jika error tentang `n_features`, artinya model kamu training pakai fitur > 1 (misal OHLCV/indikator). "
            "Ubah `build_features_for_model()` agar outputnya sesuai `model.input_shape` dan pakai scaler yang sama."
        )

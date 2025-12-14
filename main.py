import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="MarketSense", layout="wide")

# =========================
# Model loader
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"  # pastikan file ini ada di repo root (sefolder main.py)

@st.cache_resource
def load_model(path: str):
    # compile=False biar aman kalau ada custom metric/loss saat training
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error("File model tidak ditemukan. Pastikan `LSTM_CNN_model.h5` ada 1 folder dengan `main.py`.")
    st.stop()

model = load_model(str(MODEL_PATH))

# =========================
# UI
# =========================
st.title("MarketSense")
st.markdown("Masukkan ticker saham, lihat chart historis dari Yahoo Finance, lalu tampilkan prediksi. 📈")

col1, col2, col3, col4 = st.columns([2, 1.2, 1.2, 1.2])

with col1:
    nama_saham = st.text_input("Ticker saham (Yahoo Finance)", placeholder="contoh: BBCA.JK / AAPL / TSLA")

with col2:
    jenis_strategi = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))

with col3:
    jangka_prediksi = st.selectbox("Jangka waktu", ("5 hari kedepan", "10 hari kedepan"))

with col4:
    period = st.selectbox("Data historis", ("3mo", "6mo", "1y", "2y", "5y"), index=2)

STRATEGY_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}

# =========================
# Data fetch (yfinance)
# =========================
@st.cache_data(ttl=60 * 10)
def fetch_ohlc(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # pastikan kolom standar ada
    needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()
    return df

# =========================
# Build model input (TODO: ganti sesuai pipeline training kamu)
# =========================
def build_features_for_model(df: pd.DataFrame):
    """
    IMPORTANT:
    Ini masih template. Kamu harus samakan dengan pipeline training:
    - fitur apa saja (Close saja? OHLCV? indikator?)
    - scaling (MinMax/StandardScaler) yang sama
    - window length (timesteps) sesuai model.input_shape
    """
    inp = model.input_shape  # contoh: (None, 60, 5) atau (None, 60, 5, 1)

    if isinstance(inp, list):
        raise ValueError("Model kamu multi-input. Perlu builder khusus untuk tiap input.")

    if len(inp) == 3:
        _, timesteps, n_features = inp
        channels = None
    elif len(inp) == 4:
        _, timesteps, n_features, channels = inp
    else:
        raise ValueError(f"Unexpected input shape: {inp}")

    # ---- CONTOH SIMPLE: pakai Close saja ----
    # Kalau training kamu pakai OHLCV (5 fitur), ganti ini sesuai kebutuhan.
    close = df["Close"].astype(float).to_numpy()

    if len(close) < timesteps:
        raise ValueError(f"Data kurang. Butuh minimal {timesteps} bar data, tapi cuma ada {len(close)}.")

    window = close[-timesteps:]  # ambil window terakhir

    # TODO: scaling sesuai training (contoh MinMax)
    # window = (window - window.min()) / (window.max() - window.min() + 1e-9)

    # bentuk jadi (1, timesteps, n_features)
    if n_features == 1:
        X = window.reshape(1, timesteps, 1).astype(np.float32)
    else:
        # kalau model butuh banyak fitur tapi kamu baru pakai close, ini pasti mismatch
        raise ValueError(
            f"Model butuh n_features={n_features}, tapi builder ini baru menyiapkan 1 fitur (Close). "
            "Sesuaikan builder dengan fitur training kamu (misal OHLCV)."
        )

    if channels == 1:
        X = X[..., np.newaxis]  # (1, timesteps, n_features, 1)

    return X

# =========================
# Plot helpers
# =========================
def plot_stock_chart(df: pd.DataFrame, title: str, pred_series: pd.Series | None = None):
    fig = go.Figure()

    # Candlestick historis
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

    # Garis Close
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close",
        )
    )

    # Prediksi (garis ke depan)
    if pred_series is not None and not pred_series.empty:
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
# Main action
# =========================
if st.button("Submit"):
    if not nama_saham.strip():
        st.error("Ticker saham belum diisi.")
        st.stop()

    df = fetch_ohlc(nama_saham.strip(), period=period, interval="1d")
    if df.empty:
        st.error("Data tidak ditemukan. Cek ticker (contoh Indonesia biasanya pakai .JK: BBCA.JK).")
        st.stop()

    horizon = HORIZON_MAP[jangka_prediksi]
    strategy_id = STRATEGY_MAP[jenis_strategi]

    # Chart historis dulu
    st.subheader("Chart Historis (Yahoo Finance)")
    plot_stock_chart(df, f"{nama_saham.upper()} | Period: {period}")

    # Predict
    try:
        X = build_features_for_model(df)
        y_pred = model.predict(X, verbose=0)
        y_flat = np.array(y_pred).squeeze()

        # Kalau output model bukan sequence 5/10, tetap kita coba jadikan series
        if y_flat.ndim == 0:
            # single value -> repeat untuk horizon agar bisa diplot
            y_flat = np.repeat(float(y_flat), horizon)
        elif y_flat.ndim == 1:
            # kalau panjangnya beda dari horizon, tetap plot apa adanya
            pass
        else:
            raise ValueError(f"Output shape tidak didukung untuk chart: {np.array(y_pred).shape}")

        # index tanggal ke depan (business days)
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=len(y_flat))

        # TODO: kalau prediksi masih dalam skala normalisasi, lakukan inverse transform di sini
        forecast = pd.Series(y_flat.astype(float), index=future_dates)

        st.subheader("Chart dengan Forecast")
        plot_stock_chart(df, f"{nama_saham.upper()} + Forecast ({jangka_prediksi})", pred_series=forecast)

        st.caption(f"Strategy: {jenis_strategi} (id={strategy_id}) | Horizon: {horizon} hari")

        with st.expander("Nilai forecast"):
            st.dataframe(forecast.to_frame("Forecast"))

    except Exception as e:
        st.error("Gagal melakukan prediksi.")
        st.exception(e)
        st.info(
            "Kalau error bilang `n_features mismatch`, berarti model kamu training pakai fitur lebih dari 1 "
            "(misal OHLCV/indikator). Sesuaikan fungsi `build_features_for_model()` dengan pipeline training kamu."
        )

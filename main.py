import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import requests
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="MarketSense", layout="wide")

# =========================
# Config
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"

TWELVE_DATA_API_KEY = "0fa9874ef304455b944f00d57ed3473e"
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
# Twelve Data API fetch
# =========================
@st.cache_data(ttl=600)
def fetch_twelve_data(ticker: str, start, end) -> pd.DataFrame:
    """
    Fetch OHLC data from Twelve Data API
    """
    try:
        # Remove .JK suffix for Indonesian stocks
        symbol = ticker.replace(".JK", "")
        
        # Calculate outputsize (max 5000)
        days_diff = (end - start).days
        outputsize = min(days_diff + 10, 5000)
        
        # API endpoint
        url = "https://api.twelvedata.com/time_series"
        
        params = {
            "symbol": symbol,
            "interval": "1day",
            "apikey": TWELVE_DATA_API_KEY,
            "outputsize": outputsize,
            "format": "JSON",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d")
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return pd.DataFrame({"__error__": [f"API Error: Status {response.status_code}"]})
        
        data = response.json()
        
        # Check for errors
        if "status" in data and data["status"] == "error":
            error_msg = data.get("message", "Unknown error")
            return pd.DataFrame({"__error__": [f"Twelve Data Error: {error_msg}"]})
        
        # Check if data exists
        if "values" not in data or not data["values"]:
            return pd.DataFrame({"__error__": [f"No data available for {ticker}"]})
        
        # Convert to DataFrame
        df = pd.DataFrame(data["values"])
        
        # Convert columns
        df["Date"] = pd.to_datetime(df["datetime"])
        df["Open"] = pd.to_numeric(df["open"], errors="coerce")
        df["High"] = pd.to_numeric(df["high"], errors="coerce")
        df["Low"] = pd.to_numeric(df["low"], errors="coerce")
        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["volume"], errors="coerce")
        
        # Select and clean
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        return df
        
    except requests.exceptions.Timeout:
        return pd.DataFrame({"__error__": ["Request timeout. Please try again."]})
    except Exception as e:
        return pd.DataFrame({"__error__": [f"Error: {str(e)[:200]}"]})

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
# Build input for model
# =========================
def build_features_for_model(df: pd.DataFrame) -> np.ndarray:
    inp = model.input_shape

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

    # Scaling
    wmin, wmax = window.min(), window.max()
    window = (window - wmin) / (wmax - wmin + 1e-9)

    if n_features != 1:
        raise ValueError(
            f"Model butuh n_features={n_features}, tapi builder ini hanya menyiapkan 1 fitur (Close)."
        )

    X = window.reshape(1, timesteps, 1).astype(np.float32)
    if channels == 1:
        X = X[..., np.newaxis]

    return X

# =========================
# Forecast helpers
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

    # Historical price
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines+markers",
            name="Harga Historis",
            line=dict(color="#00D9FF"),
        )
    )

    # Safe zone
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
            fillcolor="rgba(0, 217, 255, 0.2)",
            line=dict(width=0),
        )
    )

    # AI trend
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=ai_series.values,
            mode="lines",
            name="AI Trend",
            line=dict(dash="dash", color="#FFD700", width=2),
        )
    )

    # Resistance / Support
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper.values,
            mode="lines",
            name="Resistance",
            line=dict(dash="dot", color="#FF4444", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower.values,
            mode="lines",
            name="Support",
            line=dict(dash="dot", color="#44FF44", width=2),
        )
    )

    fig.update_layout(
        title=f"Forecast {ticker.upper()}: {horizon} Days Ahead",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        height=580,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(x=0.01, y=0.99),
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# UI
# =========================
st.title("MarketSense")
st.markdown("🚀 AI Forecast + ATR Zone (Powered by Twelve Data API)")

# Helpful examples
with st.expander("📋 Supported Tickers"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🇺🇸 US Stocks:**
        - AAPL (Apple)
        - MSFT (Microsoft)
        - GOOGL (Google)
        - TSLA (Tesla)
        - NVDA (NVIDIA)
        - AMZN (Amazon)
        """)
    with col2:
        st.markdown("""
        **🇮🇩 Indonesian Stocks:**
        - BBCA (Bank BCA)
        - BBRI (Bank BRI)
        - TLKM (Telkom)
        - ASII (Astra)
        - UNVR (Unilever)
        
        *Note: No need for .JK suffix*
        """)

c1, c2, c3 = st.columns([2.2, 1.2, 1.2])
with c1:
    nama_saham = st.text_input("Ticker Saham", placeholder="contoh: AAPL, BBCA, MSFT")
with c2:
    strategi = st.selectbox("Strategi", ("Conservative", "Moderate", "Aggressive"))
with c3:
    horizon_label = st.selectbox("Horizon", ("5 hari kedepan", "10 hari kedepan"))

# Smart defaults: 1 year of data
default_end = datetime.now().date()
default_start = default_end - timedelta(days=365)

d1, d2 = st.columns(2)
with d1:
    start_date = st.date_input("Start Date", value=default_start)
with d2:
    end_date = st.date_input("End Date", value=default_end)

if st.button("🔮 Submit", type="primary"):
    ticker = (nama_saham or "").strip()
    if not ticker:
        st.error("❌ Ticker saham belum diisi.")
        st.stop()

    if start_date >= end_date:
        st.error("❌ Start Date harus lebih kecil dari End Date.")
        st.stop()

    horizon = HORIZON_MAP[horizon_label]
    atr_mult = ATR_MULT[strategi]

    with st.spinner(f"🔍 Fetching data for {ticker.upper()}..."):
        df = fetch_twelve_data(ticker, start_date, end_date)

    # Check for errors
    if "__error__" in df.columns:
        st.error(df["__error__"].iloc[0])
        st.info("💡 **Tips:**\n- Pastikan ticker valid (contoh: AAPL, BBCA, MSFT)\n- Coba kurangi rentang waktu\n- Untuk saham Indonesia, jangan gunakan .JK")
        st.stop()

    if df.empty:
        st.error("❌ Data kosong dari Twelve Data API.")
        st.stop()

    st.success(f"✅ Berhasil mengambil {len(df)} data points untuk {ticker.upper()}")

    # Show data preview
    with st.expander("📊 Data Preview"):
        st.dataframe(df.tail(10))

    # Calculate ATR
    atr = compute_atr(df, period=14)
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0

    try:
        with st.spinner("🤖 Running AI prediction..."):
            X = build_features_for_model(df)
            y_pred = model.predict(X, verbose=0)

            last_close = float(df["Close"].iloc[-1])
            ai_trend = make_ai_trend(y_pred, last_close=last_close, horizon=horizon)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Close", f"${last_close:.2f}")
        with col2:
            predicted_price = ai_trend[-1]
            change_pct = ((predicted_price - last_close) / last_close) * 100
            st.metric(f"Predicted ({horizon}d)", f"${predicted_price:.2f}", f"{change_pct:+.2f}%")
        with col3:
            st.metric("ATR (14)", f"${atr_last:.2f}")
        with col4:
            st.metric("Strategy", strategi)

        # Plot
        plot_forecast_like_screenshot(
            df=df,
            ticker=ticker,
            ai_trend=ai_trend,
            atr_last=atr_last,
            atr_mult=atr_mult,
            horizon=horizon,
        )

        # Additional info
        st.info(f"📈 **Resistance Level:** ${(ai_trend[-1] + atr_last * atr_mult):.2f} | "
                f"📉 **Support Level:** ${(ai_trend[-1] - atr_last * atr_mult):.2f}")

    except Exception as e:
        st.error("❌ Gagal membuat forecast.")
        st.exception(e)
        st.info(
            "💡 Kalau error `n_features mismatch`, berarti model training pakai fitur > 1 (OHLCV/indikator). "
            "Ubah `build_features_for_model()` agar sesuai `model.input_shape` dan scaling training."
        )

# Footer
st.markdown("---")
st.caption("Powered by Twelve Data API | Model: LSTM+CNN | Strategy: ATR-based Risk Management")
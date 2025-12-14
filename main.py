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
# Build input for model - EXACT TRAINING FEATURES
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

    if len(df) < timesteps + 30:  # Extra buffer for indicators
        raise ValueError(f"Data kurang. Butuh minimal {timesteps + 30} bar, tapi hanya ada {len(df)}.")

    # Build exact training features
    df_features = df.copy()
    
    # Convert to float
    df_features["Close"] = df_features["Close"].astype(float)
    df_features["Volume"] = df_features["Volume"].astype(float)
    df_features["High"] = df_features["High"].astype(float)
    df_features["Low"] = df_features["Low"].astype(float)
    
    # 1. Close (already have it)
    
    # 2. Volume (already have it)
    
    # 3. LogReturn
    df_features["LogReturn"] = np.log(df_features["Close"] / df_features["Close"].shift(1))
    
    # 4. LogReturn_Lag1
    df_features["LogReturn_Lag1"] = df_features["LogReturn"].shift(1)
    
    # 5. LogReturn_Lag2
    df_features["LogReturn_Lag2"] = df_features["LogReturn"].shift(2)
    
    # 6. LogReturn_Lag3
    df_features["LogReturn_Lag3"] = df_features["LogReturn"].shift(3)
    
    # 7. ATR_14
    high = df_features["High"]
    low = df_features["Low"]
    close = df_features["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df_features["ATR_14"] = tr.rolling(14).mean()
    
    # 8. RSI_14
    delta = df_features["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df_features["RSI_14"] = 100 - (100 / (1 + rs))
    
    # 9. MACD
    ema_12 = df_features["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df_features["Close"].ewm(span=26, adjust=False).mean()
    df_features["MACD"] = ema_12 - ema_26
    
    # 10. VolChange
    df_features["VolChange"] = df_features["Volume"].pct_change()
    
    # Select exact features in exact order
    feature_cols = ['Close', 'Volume', 'LogReturn', 'LogReturn_Lag1', 'LogReturn_Lag2', 'LogReturn_Lag3',
                    'ATR_14', 'RSI_14', 'MACD', 'VolChange']
    
    df_features = df_features[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df_features) < timesteps:
        raise ValueError(
            f"Setelah menghitung indikator, data tersisa {len(df_features)} bar. "
            f"Butuh minimal {timesteps} bar. Gunakan start date lebih awal (minimal 1 tahun data)."
        )
    
    # Get last window
    window = df_features.tail(timesteps).values
    
    # Min-Max scaling per feature
    feature_mins = window.min(axis=0)
    feature_maxs = window.max(axis=0)
    window_scaled = (window - feature_mins) / (feature_maxs - feature_mins + 1e-9)
    
    # Replace any remaining nan/inf with 0
    window_scaled = np.nan_to_num(window_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check if model expects correct number of features
    if n_features != 10:
        raise ValueError(
            f"Model butuh n_features={n_features}, tapi builder menyiapkan 10 fitur. "
            f"Model input shape: {inp}"
        )
    
    X = window_scaled.reshape(1, timesteps, 10).astype(np.float32)
    if channels == 1:
        X = X[..., np.newaxis]  # (1, timesteps, 10, 1)

    return X

# =========================
# Forecast helpers
# =========================
def make_ai_trend(y_pred: np.ndarray, last_close: float, horizon: int) -> np.ndarray:
    """
    Convert model predictions to price trend
    Model predicts: [Target_High_Ret, Target_Low_Ret]
    """
    y = np.array(y_pred).squeeze()
    
    # If model outputs 2 values (high_ret, low_ret), use average
    if y.ndim == 1 and len(y) == 2:
        avg_return = (y[0] + y[1]) / 2
        # Convert return to price
        end_price = last_close * (1 + avg_return)
        return np.linspace(last_close, end_price, num=horizon + 1)[1:]
    
    # If model outputs single value
    if y.ndim == 0 or (y.ndim == 1 and len(y) == 1):
        end_val = float(y[0]) if y.ndim == 1 else float(y)
        end_price = last_close * (1 + end_val)
        return np.linspace(last_close, end_price, num=horizon + 1)[1:]
    
    # If model outputs sequence
    if y.ndim == 1 and len(y) > 2:
        if len(y) == horizon:
            return y.astype(float)
        # Resample to horizon
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
            line=dict(color="#00D9FF", width=2),
            marker=dict(size=4),
        )
    )

    # Safe zone (fill)
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
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
            hoverinfo='skip',
        )
    )

    # AI trend
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=ai_series.values,
            mode="lines+markers",
            name="AI Trend",
            line=dict(dash="dash", color="#FFD700", width=3),
            marker=dict(size=6),
        )
    )

    # Resistance
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper.values,
            mode="lines",
            name="Resistance",
            line=dict(dash="dot", color="#FF4444", width=2),
        )
    )
    
    # Support
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
        title=f"📈 Forecast {ticker.upper()}: {horizon} Days Ahead",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        height=600,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            x=0.01, 
            y=0.99,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# UI
# =========================
st.title("📊 MarketSense")
st.markdown("🚀 **AI-Powered Stock Forecast** | Powered by Twelve Data API")

# Helpful examples
with st.expander("📋 Supported Tickers & Examples"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🇺🇸 US Stocks:**
        - `AAPL` - Apple Inc.
        - `MSFT` - Microsoft
        - `GOOGL` - Alphabet (Google)
        - `TSLA` - Tesla
        - `NVDA` - NVIDIA
        - `AMZN` - Amazon
        - `META` - Meta (Facebook)
        """)
    with col2:
        st.markdown("""
        **🇮🇩 Indonesian Stocks:**
        - `BBCA` - Bank BCA
        - `BBRI` - Bank BRI
        - `TLKM` - Telkom Indonesia
        - `ASII` - Astra International
        - `UNVR` - Unilever Indonesia
        
        💡 *No .JK suffix needed*
        """)

c1, c2, c3 = st.columns([2.2, 1.2, 1.2])
with c1:
    nama_saham = st.text_input("Ticker Saham", placeholder="contoh: AAPL, BBCA, MSFT")
with c2:
    strategi = st.selectbox("Strategi Risk", ("Conservative", "Moderate", "Aggressive"))
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

st.info("💡 **Tip:** Gunakan minimal 1 tahun data untuk hasil prediksi optimal")

if st.button("🔮 Analyze & Predict", type="primary", use_container_width=True):
    ticker = (nama_saham or "").strip()
    if not ticker:
        st.error("❌ Ticker saham belum diisi.")
        st.stop()

    if start_date >= end_date:
        st.error("❌ Start Date harus lebih kecil dari End Date.")
        st.stop()
    
    # Check minimum date range
    days_diff = (end_date - start_date).days
    if days_diff < 180:
        st.warning("⚠️ Rentang waktu terlalu pendek. Minimal 6 bulan untuk hasil optimal.")

    horizon = HORIZON_MAP[horizon_label]
    atr_mult = ATR_MULT[strategi]

    with st.spinner(f"🔍 Fetching data for {ticker.upper()} from Twelve Data API..."):
        df = fetch_twelve_data(ticker, start_date, end_date)

    # Check for errors
    if "__error__" in df.columns:
        st.error(df["__error__"].iloc[0])
        st.info("💡 **Tips:**\n- Pastikan ticker valid (contoh: AAPL, BBCA, MSFT)\n- Coba kurangi rentang waktu\n- Untuk saham Indonesia, jangan gunakan .JK")
        st.stop()

    if df.empty:
        st.error("❌ Data kosong dari Twelve Data API.")
        st.stop()

    st.success(f"✅ Berhasil mengambil **{len(df)}** data points untuk **{ticker.upper()}**")

    # Show data preview
    with st.expander("📊 Data Preview (Last 10 rows)"):
        st.dataframe(df.tail(10), use_container_width=True)

    # Calculate ATR
    atr = compute_atr(df, period=14)
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0

    try:
        with st.spinner("🤖 Running AI prediction model..."):
            X = build_features_for_model(df)
            y_pred = model.predict(X, verbose=0)

            last_close = float(df["Close"].iloc[-1])
            ai_trend = make_ai_trend(y_pred, last_close=last_close, horizon=horizon)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Last Close", f"${last_close:.2f}")
        with col2:
            predicted_price = ai_trend[-1]
            change_pct = ((predicted_price - last_close) / last_close) * 100
            st.metric(f"🎯 Predicted ({horizon}d)", f"${predicted_price:.2f}", f"{change_pct:+.2f}%")
        with col3:
            st.metric("📊 ATR (14)", f"${atr_last:.2f}")
        with col4:
            st.metric("🛡️ Strategy", strategi)

        # Plot
        plot_forecast_like_screenshot(
            df=df,
            ticker=ticker,
            ai_trend=ai_trend,
            atr_last=atr_last,
            atr_mult=atr_mult,
            horizon=horizon,
        )

        # Key levels
        resistance = ai_trend[-1] + (atr_last * atr_mult)
        support = ai_trend[-1] - (atr_last * atr_mult)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"📈 **Resistance Level:** ${resistance:.2f}")
        with col2:
            st.error(f"📉 **Support Level:** ${support:.2f}")
        
        # Trading signal
        if change_pct > 2:
            st.success("🟢 **Signal:** BULLISH - Potensi kenaikan harga")
        elif change_pct < -2:
            st.error("🔴 **Signal:** BEARISH - Potensi penurunan harga")
        else:
            st.info("🟡 **Signal:** NEUTRAL - Sideways/konsolidasi")

    except Exception as e:
        st.error("❌ Gagal membuat forecast.")
        st.exception(e)
        st.info(
            "💡 **Troubleshooting:**\n"
            "- Pastikan rentang tanggal cukup panjang (minimal 1 tahun)\n"
            "- Model membutuhkan 10 fitur dengan timesteps tertentu\n"
            "- Coba ticker lain atau rentang waktu berbeda"
        )

# Footer
st.markdown("---")
st.caption("⚡ Powered by Twelve Data API | 🤖 Model: LSTM+CNN | 📊 Strategy: ATR-based Risk Management")
st.caption("⚠️ Disclaimer: Prediksi ini hanya untuk referensi. Lakukan riset mandiri sebelum trading.")
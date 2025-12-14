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
# Config
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"

TWELVE_DATA_API_KEY = "0fa9874ef304455b944f00d57ed3473e"  # <-- consider moving to st.secrets
HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

# A simple IDX ticker list used by your currency detector
INDONESIAN_TICKERS = [
    "BBCA", "BBRI", "BBNI", "BMRI", "TLKM", "ASII", "UNVR", "ICBP", "INDF",
    "KLBF", "GGRM", "HMSP", "SMGR", "JSMR", "PTBA", "ADRO", "ITMG", "ANTM",
    "INCO", "EXCL", "GOTO",
]

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
# FX: USD -> IDR
# =========================
@st.cache_data(ttl=3600)
def fetch_usd_idr_rate() -> float:
    """
    Fetch USD/IDR exchange rate from Twelve Data.
    Cached 1 hour.
    """
    url = "https://api.twelvedata.com/exchange_rate"
    params = {"symbol": "USD/IDR", "apikey": TWELVE_DATA_API_KEY}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    # Twelve Data error format
    if isinstance(data, dict) and data.get("status") == "error":
        raise ValueError(data.get("message", "Failed to fetch USD/IDR rate"))

    if "rate" not in data:
        raise ValueError("Failed to fetch USD/IDR rate (missing 'rate' field)")

    return float(data["rate"])

# =========================
# Currency Detection
# =========================
def detect_currency(ticker: str) -> tuple[str, str]:
    """
    Detect whether ticker is Indonesian (IDR) or US (USD).
    Returns (currency_symbol, currency_code).
    """
    ticker_clean = ticker.upper().replace(".JK", "")
    if ticker_clean in INDONESIAN_TICKERS:
        return ("Rp", "IDR")
    return ("$", "USD")

# =========================
# Twelve Data API fetch (OHLCV)
# =========================
@st.cache_data(ttl=600)
def fetch_twelve_data(ticker: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetch OHLC data from Twelve Data API for last N days.
    Converts USD -> IDR for Indonesian tickers (based on INDONESIAN_TICKERS list).
    """
    try:
        symbol = ticker.upper().replace(".JK", "")  # Your UX: no .JK needed

        outputsize = min(days_back + 30, 5000)  # buffer for indicators
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            "apikey": TWELVE_DATA_API_KEY,
            "outputsize": outputsize,
            "format": "JSON",
            "order": "ASC",
        }

        response = requests.get(url, params=params, timeout=20)
        if response.status_code != 200:
            return pd.DataFrame({"__error__": [f"API Error: Status {response.status_code}"]})

        data = response.json()

        # Handle Twelve Data errors
        if isinstance(data, dict) and data.get("status") == "error":
            return pd.DataFrame({"__error__": [f"Twelve Data Error: {data.get('message', 'Unknown error')}"]})

        if "values" not in data or not data["values"]:
            return pd.DataFrame({"__error__": [f"No data available for {ticker}"]})

        df = pd.DataFrame(data["values"])

        # Normalize
        df["Date"] = pd.to_datetime(df["datetime"])
        df["Open"] = pd.to_numeric(df["open"], errors="coerce")
        df["High"] = pd.to_numeric(df["high"], errors="coerce")
        df["Low"] = pd.to_numeric(df["low"], errors="coerce")
        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df.get("volume", np.nan), errors="coerce")

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)

        # =========================
        # USD -> IDR conversion for Indonesian tickers
        # =========================
        if symbol in INDONESIAN_TICKERS:
            usd_idr = fetch_usd_idr_rate()
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = df[col] * usd_idr

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
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()

# =========================
# Build model input (your feature pipeline)
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

    if len(df) < timesteps + 30:
        raise ValueError(f"Data kurang. Butuh minimal {timesteps + 30} bar, tapi hanya ada {len(df)}.")

    d = df.copy()
    d["Close"] = d["Close"].astype(float)
    d["Volume"] = d["Volume"].astype(float)
    d["High"] = d["High"].astype(float)
    d["Low"] = d["Low"].astype(float)

    # Features
    d["LogReturn"] = np.log(d["Close"] / d["Close"].shift(1))
    d["LogReturn_Lag1"] = d["LogReturn"].shift(1)
    d["LogReturn_Lag2"] = d["LogReturn"].shift(2)
    d["LogReturn_Lag3"] = d["LogReturn"].shift(3)

    # ATR_14 (same as compute_atr but inline)
    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [(d["High"] - d["Low"]).abs(), (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()

    # RSI_14
    delta = d["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26)
    ema_12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema_12 - ema_26

    # VolChange
    d["VolChange"] = d["Volume"].pct_change()

    feature_cols = [
        "Close", "Volume", "LogReturn", "LogReturn_Lag1", "LogReturn_Lag2", "LogReturn_Lag3",
        "ATR_14", "RSI_14", "MACD", "VolChange"
    ]

    d = d[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(d) < timesteps:
        raise ValueError(
            f"Setelah menghitung indikator, data tersisa {len(d)} bar. "
            f"Butuh minimal {timesteps} bar. Ambil data lebih panjang."
        )

    window = d.tail(timesteps).values

    # Per-window min-max scaling (match your current approach)
    mins = window.min(axis=0)
    maxs = window.max(axis=0)
    window_scaled = (window - mins) / (maxs - mins + 1e-9)
    window_scaled = np.nan_to_num(window_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    if n_features != 10:
        raise ValueError(f"Model butuh n_features={n_features}, tapi builder menyiapkan 10 fitur. Model shape: {inp}")

    X = window_scaled.reshape(1, timesteps, 10).astype(np.float32)
    if channels == 1:
        X = X[..., np.newaxis]

    return X

# =========================
# Forecast helpers
# =========================
def make_ai_trend(y_pred: np.ndarray, last_close: float, horizon: int) -> np.ndarray:
    """
    Convert model predictions to price trend.
    If output is (high_ret, low_ret): use average return and project linearly.
    """
    y = np.array(y_pred).squeeze()

    if y.ndim == 1 and len(y) == 2:
        avg_return = (float(y[0]) + float(y[1])) / 2.0
        end_price = last_close * (1.0 + avg_return)
        return np.linspace(last_close, end_price, num=horizon + 1)[1:]

    if y.ndim == 0:
        end_price = last_close * (1.0 + float(y))
        return np.linspace(last_close, end_price, num=horizon + 1)[1:]

    if y.ndim == 1 and len(y) == 1:
        end_price = last_close * (1.0 + float(y[0]))
        return np.linspace(last_close, end_price, num=horizon + 1)[1:]

    if y.ndim == 1 and len(y) > 2:
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
    currency_symbol: str = "$",
):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    last_date = df["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    ai_series = pd.Series(ai_trend, index=future_dates)
    band = (atr_last * atr_mult) if atr_last > 0 else 0.0
    upper = ai_series + band
    lower = ai_series - band

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Close"],
            mode="lines+markers",
            name="Harga Historis",
            line=dict(color="#00D9FF", width=2),
            marker=dict(size=4),
        )
    )

    # Safe zone fill
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=upper.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=lower.values,
            mode="lines",
            fill="tonexty",
            name=f"Zona Aman (AI + ATR {atr_mult:.1f}x)",
            fillcolor="rgba(0, 217, 255, 0.2)",
            line=dict(width=0),
            hoverinfo="skip",
        )
    )

    # AI Trend
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=ai_series.values,
            mode="lines+markers",
            name="AI Trend (Daily Adjusted)",
            line=dict(dash="dash", color="#FFD700", width=3),
            marker=dict(size=6),
        )
    )

    # Resistance/Support
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=upper.values,
            mode="lines",
            name="Resistance",
            line=dict(dash="dot", color="#FF4444", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=lower.values,
            mode="lines",
            name="Support",
            line=dict(dash="dot", color="#44FF44", width=2),
        )
    )

    fig.update_layout(
        title=f"Final Forecast {ticker.upper()}: AI Horizon Adjusted ( / {horizon})",
        xaxis_title="Tanggal",
        yaxis_title=f"Harga ({currency_symbol})",
        height=600,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# UI
# =========================
st.title("📊 MarketSense")
st.markdown("🚀 **AI Forecast + ATR Zone** | Powered by Twelve Data API")

c1, c2, c3 = st.columns(3)
with c1:
    nama_saham = st.text_input("📌 Ticker Saham", placeholder="contoh: BBCA, BBNI, TLKM, AAPL")
with c2:
    strategi = st.selectbox("🛡️ Strategi Risk", ("Conservative", "Moderate", "Aggressive"))
with c3:
    horizon_label = st.selectbox("🎯 Horizon", ("5 hari kedepan", "10 hari kedepan"))

st.info("💡 Auto-fetch: sistem mengambil 1 tahun data historis terbaru.")

if st.button("🔮 Analyze & Predict", type="primary", use_container_width=True):
    ticker = (nama_saham or "").strip().upper()
    if not ticker:
        st.error("❌ Ticker saham belum diisi.")
        st.stop()

    horizon = HORIZON_MAP[horizon_label]
    atr_mult = ATR_MULT[strategi]

    currency_symbol, currency_code = detect_currency(ticker)

    with st.spinner(f"🔍 Fetching data for {ticker}..."):
        df = fetch_twelve_data(ticker, days_back=365)

    if "__error__" in df.columns:
        st.error(df["__error__"].iloc[0])
        st.info("Tips: Untuk saham Indonesia cukup pakai kode IDX tanpa .JK (contoh: BBCA, BBNI, TLKM).")
        st.stop()

    if df.empty:
        st.error("❌ Data kosong dari Twelve Data API.")
        st.stop()

    # Show FX rate if IDR conversion happened
    if ticker.replace(".JK", "") in INDONESIAN_TICKERS:
        try:
            usd_idr = fetch_usd_idr_rate()
            st.caption(f"FX used: 1 USD = {usd_idr:,.2f} IDR (via Twelve Data)")
        except Exception:
            pass

    atr = compute_atr(df, period=14)
    atr_last = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0.0

    try:
        with st.spinner("🤖 Running AI model..."):
            X = build_features_for_model(df)
            y_pred = model.predict(X, verbose=0)

            last_close = float(df["Close"].iloc[-1])
            ai_trend = make_ai_trend(y_pred, last_close=last_close, horizon=horizon)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        def fmt_price(x: float) -> str:
            return f"{currency_symbol}{x:,.0f}" if currency_code == "IDR" else f"{currency_symbol}{x:.2f}"

        with col1:
            st.metric("💰 Last Close", fmt_price(last_close))
        with col2:
            predicted_price = float(ai_trend[-1])
            change_pct = ((predicted_price - last_close) / last_close) * 100.0
            st.metric(f"🎯 Predicted ({horizon}d)", fmt_price(predicted_price), f"{change_pct:+.2f}%")
        with col3:
            st.metric("📊 ATR (14)", fmt_price(atr_last))
        with col4:
            st.metric("🛡️ Strategy", strategi)

        plot_forecast_like_screenshot(
            df=df,
            ticker=ticker,
            ai_trend=ai_trend,
            atr_last=atr_last,
            atr_mult=atr_mult,
            horizon=horizon,
            currency_symbol=currency_symbol,
        )

        # Levels
        resistance = float(ai_trend[-1]) + (atr_last * atr_mult)
        support = float(ai_trend[-1]) - (atr_last * atr_mult)

        cL, cR = st.columns(2)
        with cL:
            st.success(f"📈 **Resistance:** {fmt_price(resistance)}")
        with cR:
            st.error(f"📉 **Support:** {fmt_price(support)}")

        # Signal
        if change_pct > 2:
            st.success("🟢 **Signal:** BULLISH")
        elif change_pct < -2:
            st.error("🔴 **Signal:** BEARISH")
        else:
            st.info("🟡 **Signal:** NEUTRAL")

    except Exception as e:
        st.error("❌ Gagal membuat forecast.")
        st.exception(e)
        st.info(
            "Troubleshooting:\n"
            "- Pastikan data historis cukup panjang\n"
            "- Pastikan model input_shape cocok dengan 10 fitur yang dibangun\n"
            "- Kalau modelmu training pakai scaler khusus, samakan di builder"
        )

st.markdown("---")
st.caption("⚠️ Disclaimer: Prediksi ini hanya untuk referensi. Lakukan riset mandiri sebelum trading.")

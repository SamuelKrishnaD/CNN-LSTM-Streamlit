import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="MarketSense IDX", layout="wide")

# =========================
# Config
# =========================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.keras" / "LSTM_CNN_model.keras"

HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}
ATR_MULT = {"Conservative": 0.8, "Moderate": 1.0, "Aggressive": 1.3}

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path, compile=False)

if not MODEL_PATH.exists():
    st.error(f"Model file tidak ditemukan di: {MODEL_PATH}")
    st.info("Pastikan folder `LSTM_CNN_model.keras` ada dan berisi file `LSTM_CNN_model.keras`")
    st.stop()

try:
    model = load_model_cached(str(MODEL_PATH))
except Exception as e:
    st.error("Gagal load model .keras")
    st.exception(e)
    st.stop()

# =========================
# YFinance fetch for IDX stocks - AUTO DATE RANGE
# =========================
@st.cache_data(ttl=600)
def fetch_idx_data(ticker: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetch OHLC data from Yahoo Finance for Indonesian stocks
    Automatically fetches last N days of data
    """
    try:
        # Add .JK suffix for Indonesian stocks if not present
        symbol = ticker.upper()
        if not symbol.endswith(".JK"):
            symbol = f"{symbol}.JK"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Download data
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df is None or df.empty:
            return pd.DataFrame({"__error__": [f"No data available for {ticker}"]})
        
        # Reset index
        df = df.reset_index()
        
        # Flatten multiindex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Normalize date column
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        
        # Check required columns
        needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            return pd.DataFrame({"__error__": [f"Missing required columns for {ticker}"]})
        
        # Select and clean
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        return df
        
    except Exception as e:
        return pd.DataFrame({"__error__": [f"Error fetching {ticker}: {str(e)[:200]}"]})

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
        raise ValueError(
            f"⚠️ Data tidak cukup!\n"
            f"- Data tersedia: {len(df)} hari\n"
            f"- Dibutuhkan: {timesteps + 30} hari\n"
            f"Model membutuhkan {timesteps} timesteps + 30 hari untuk indikator"
        )

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
            f"⚠️ Setelah menghitung indikator teknikal, data tersisa hanya {len(df_features)} hari.\n"
            f"Model membutuhkan minimal {timesteps} hari.\n"
            f"**Solusi:** Gunakan periode data lebih panjang (minimal 1 tahun)"
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

def plot_forecast_idx(
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
        title=f"📈 Forecast {ticker.upper()}: {horizon} Hari Kedepan",
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
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
# UI - IDX ONLY
# =========================
st.title("📊 MarketSense IDX")
st.markdown("🇮🇩 **AI-Powered IDX Stock Forecast** | Powered by Yahoo Finance")

# IDX stock list
with st.expander("📋 Daftar Saham IDX yang Tersedia"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🏦 Perbankan:**
        - `BBCA` - Bank BCA
        - `BBRI` - Bank BRI
        - `BMRI` - Bank Mandiri
        - `BBNI` - Bank BNI
        - `BRIS` - Bank BRI Syariah
        """)
    with col2:
        st.markdown("""
        **📱 Teknologi & Telkom:**
        - `TLKM` - Telkom Indonesia
        - `EXCL` - XL Axiata
        - `GOTO` - GoTo Gojek Tokopedia
        - `ISAT` - Indosat Ooredoo
        """)
    with col3:
        st.markdown("""
        **🏭 Industri & Konsumer:**
        - `ASII` - Astra International
        - `UNVR` - Unilever Indonesia
        - `ICBP` - Indofood CBP
        - `INDF` - Indofood Sukses Makmur
        """)
    
    st.markdown("""
    **💡 Tips:** Cukup ketik kode saham tanpa suffix .JK (contoh: `BBCA`, `TLKM`, `ASII`)
    """)

# Simple 3-input form
c1, c2, c3 = st.columns(3)
with c1:
    nama_saham = st.text_input("📌 Kode Saham IDX", placeholder="contoh: BBCA, TLKM, ASII")
with c2:
    strategi = st.selectbox("🛡️ Strategi Risk", ("Conservative", "Moderate", "Aggressive"))
with c3:
    horizon_label = st.selectbox("🎯 Horizon", ("5 hari kedepan", "10 hari kedepan"))

st.info("💡 **Auto-fetch:** Sistem akan otomatis mengambil 1 tahun data historis terbaru dari Yahoo Finance")

if st.button("🔮 Analyze & Predict", type="primary", use_container_width=True):
    ticker = (nama_saham or "").strip()
    if not ticker:
        st.error("❌ Kode saham belum diisi.")
        st.stop()

    horizon = HORIZON_MAP[horizon_label]
    atr_mult = ATR_MULT[strategi]

    with st.spinner(f"🔍 Fetching 1 year data for {ticker.upper()}.JK dari Yahoo Finance..."):
        df = fetch_idx_data(ticker, days_back=365)

    # Check for errors
    if "__error__" in df.columns:
        st.error(df["__error__"].iloc[0])
        st.info("💡 **Tips:**\n- Pastikan kode saham IDX valid (contoh: BBCA, TLKM, ASII)\n- Jangan tambahkan .JK di input\n- Coba saham lain jika data tidak tersedia")
        st.stop()

    if df.empty:
        st.error("❌ Data kosong dari Yahoo Finance.")
        st.stop()

    st.success(f"✅ Berhasil mengambil **{len(df)}** data points untuk **{ticker.upper()}.JK**")

    # Show data preview
    with st.expander("📊 Data Preview (10 Hari Terakhir)"):
        preview_df = df.tail(10).copy()
        preview_df['Close'] = preview_df['Close'].apply(lambda x: f"Rp {x:,.0f}")
        preview_df['Volume'] = preview_df['Volume'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(preview_df, use_container_width=True)

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
            st.metric("💰 Harga Terakhir", f"Rp {last_close:,.0f}")
        with col2:
            predicted_price = ai_trend[-1]
            change_pct = ((predicted_price - last_close) / last_close) * 100
            st.metric(f"🎯 Prediksi ({horizon}h)", f"Rp {predicted_price:,.0f}", f"{change_pct:+.2f}%")
        with col3:
            st.metric("📊 ATR (14)", f"Rp {atr_last:,.0f}")
        with col4:
            st.metric("🛡️ Strategy", strategi)

        # Plot
        plot_forecast_idx(
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
            st.success(f"📈 **Resistance Level:** Rp {resistance:,.0f}")
        with col2:
            st.error(f"📉 **Support Level:** Rp {support:,.0f}")
        
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
            "- Model membutuhkan minimal 1 tahun data historis\n"
            "- Coba saham lain atau tunggu beberapa menit\n"
            "- Pastikan saham aktif diperdagangkan di IDX"
        )

# Footer
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance | 🤖 Model: LSTM+CNN (.keras) | 📊 Strategy: ATR-based Risk Management")
st.caption("⚠️ Disclaimer: Prediksi ini hanya untuk referensi. Lakukan riset mandiri sebelum trading.")
st.caption("🇮🇩 Khusus untuk saham Bursa Efek Indonesia (IDX)")
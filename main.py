# main.py
import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="MarketSense", layout="centered")

# ---------------------------
# Paths (safe for Streamlit Cloud)
# ---------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "LSTM_CNN_model.h5"   # put the .h5 in the SAME folder as this file

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model(path: str):
    # compile=False prevents errors if your .h5 has custom metrics/loss not needed for inference
    return tf.keras.models.load_model(path, compile=False)

st.title("MarketSense")
st.markdown("Masukkan saham + strategi + horizon prediksi, lalu jalankan modelnya. 📈")

# Debug info (helps you on Streamlit Cloud)
with st.expander("Debug: cek file & environment"):
    st.write("App directory:", str(APP_DIR))
    try:
        st.write("Files:", [p.name for p in APP_DIR.iterdir()])
    except Exception as e:
        st.write("Cannot list directory:", e)

# Ensure model exists
if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Fix: upload/commit `LSTM_CNN_model.h5` into the same folder as `main.py`.")
    st.stop()

# Load model safely
try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error("Failed to load model.")
    st.exception(e)
    st.info("If you used custom loss/metrics/layers, you may need `custom_objects`.")
    st.stop()

# Show model shapes
with st.expander("Info model"):
    st.write("Model input shape:", model.input_shape)
    st.write("Model output shape:", model.output_shape)

# ---------------------------
# UI Inputs
# ---------------------------
nama_saham = st.text_input("Sebutkan nama sahamnya", placeholder="contoh: BBCA / AAPL / TSLA")

jenis_strategi = st.selectbox(
    "Pilih jenis Strategi",
    ("Conservative", "Moderate", "Aggressive")
)

jangka_prediksi = st.selectbox(
    "Pilih jangka waktu",
    ("5 hari kedepan", "10 hari kedepan")
)

STRATEGY_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}
HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}

# ---------------------------
# Build model input (dummy, but SHAPE-CORRECT)
# Replace this later with your real pipeline:
# load OHLCV -> scale -> create window -> shape to model.input_shape
# ---------------------------
def build_features_for_model(ticker: str, strategy: str, horizon_label: str):
    horizon = HORIZON_MAP[horizon_label]
    strategy_id = STRATEGY_MAP[strategy]

    inp = model.input_shape
    # Common patterns:
    # (None, timesteps, features)
    # (None, timesteps, features, 1)
    if isinstance(inp, list):
        # Multi-input model (rare). You must build each input separately.
        raise ValueError(f"Your model has multiple inputs: {inp}. Need custom input builder.")

    if len(inp) == 3:
        _, timesteps, n_features = inp
        X = np.random.rand(1, timesteps, n_features).astype(np.float32)
    elif len(inp) == 4:
        _, timesteps, n_features, channels = inp
        if channels != 1:
            raise ValueError(f"Expected channels=1 but got {channels}. Adjust input builder.")
        X = np.random.rand(1, timesteps, n_features, 1).astype(np.float32)
    else:
        raise ValueError(f"Unexpected model input shape: {inp}")

    return X, horizon, strategy_id

# ---------------------------
# Predict
# ---------------------------
if st.button("Submit"):
    if not nama_saham.strip():
        st.error("Nama sahamnya belum diisi.")
        st.stop()

    try:
        X, horizon, strategy_id = build_features_for_model(nama_saham, jenis_strategi, jangka_prediksi)

        # Predict
        y_pred = model.predict(X, verbose=0)

        st.success("Prediksi berhasil dibuat ✅")

        st.subheader("Output Model")
        st.write(y_pred)

        # Nice display for 1D sequences
        y_flat = np.array(y_pred).squeeze()
        if y_flat.ndim == 1 and y_flat.size in (5, 10):
            st.line_chart(y_flat)

        st.caption(
            f"Ticker: {nama_saham} | Strategy: {jenis_strategi} ({strategy_id}) | Horizon: {horizon} hari"
        )

    except Exception as e:
        st.error("Terjadi error saat prediksi.")
        st.exception(e)

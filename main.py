import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="MarketSense", layout="centered")

# ---------------------------
# 1) Load model (cached)
# ---------------------------
@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

model = load_model("./LSTM_CNN_model.h5")  # TODO: rename to your file

st.title("MarketSense")
st.markdown("Masukkan saham + strategi + horizon prediksi, lalu jalankan modelnya. 📈")

# ---------------------------
# 2) UI Inputs
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

# Optional: show model input shape so you know what to feed it
with st.expander("Lihat info model"):
    st.write("Model input shape:", model.input_shape)
    st.write("Model output shape:", model.output_shape)

# ---------------------------
# 3) Helper: map UI -> numeric
# ---------------------------
STRATEGY_MAP = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}

HORIZON_MAP = {"5 hari kedepan": 5, "10 hari kedepan": 10}

# ---------------------------
# 4) Build model input (IMPORTANT)
# ---------------------------
def build_features_for_model(ticker: str, strategy: str, horizon_label: str):
    """
    Return X shaped exactly like model expects.
    You MUST implement your real preprocessing here.
    """
    horizon = HORIZON_MAP[horizon_label]
    strategy_id = STRATEGY_MAP[strategy]

    # TODO A:
    # 1) Load your recent time-series window for this ticker (from CSV/API/database)
    # 2) Apply the SAME scaling/normalization used during training (MinMaxScaler, StandardScaler, etc.)
    # 3) Construct tensor shape exactly like training input

    # ---- TEMP DUMMY EXAMPLE (replace) ----
    # Suppose your model expects (batch, timesteps, features)
    # Example: (None, 60, 5) meaning 60 days window, 5 features (OHLCV)
    timesteps = model.input_shape[1]
    n_features = model.input_shape[2]

    # Dummy data just to show shape works:
    X = np.random.rand(1, timesteps, n_features).astype(np.float32)

    # If you also trained with extra features like strategy_id/horizon,
    # you might append them into feature channels or use multi-input model.
    # For single-input CNN-LSTM, you generally bake them into features if used.
    # -------------------------------------

    return X, horizon, strategy_id

# ---------------------------
# 5) Predict
# ---------------------------
if st.button("Submit"):
    if not nama_saham.strip():
        st.error("Nama sahamnya belum diisi.")
    else:
        try:
            X, horizon, strategy_id = build_features_for_model(nama_saham, jenis_strategi, jangka_prediksi)

            # Predict
            y_pred = model.predict(X, verbose=0)

            st.success("Prediksi berhasil dibuat ✅")

            # Display output nicely
            st.subheader("Output Model")
            st.write(y_pred)

            # If y_pred is a sequence (e.g., 5 or 10 future steps), show as chart:
            y_flat = np.array(y_pred).squeeze()
            if y_flat.ndim == 1 and y_flat.size in (5, 10):
                st.line_chart(y_flat)

            # Strategy/horizon info
            st.caption(f"Ticker: {nama_saham} | Strategy: {jenis_strategi} ({strategy_id}) | Horizon: {horizon} hari")

        except Exception as e:
            st.exception(e)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import base64
import time

# ----------------------------
# Helpers: background images
# ----------------------------
def get_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_base64("adds/stonk.jpg")
page_bg_color = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

sidebar_base64 = get_base64('adds/stonk_sidebar.png')
sidebar_bg_image = f"""
<style>
[data-testid="stSidebar"] {{
    background-image: url("data:image/jpeg;base64,{sidebar_base64}");
    background-size: cover;
    background-position: center;
}}
</style>
"""
st.markdown(sidebar_bg_image, unsafe_allow_html=True)

# Sidebar input styling
st.markdown("""
<style>
[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #1E1E1E;
    color: white;
    border-radius: 8px;
    padding: 8px;
    border: 1px solid #555;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    background-color: #1E1E1E;
    color: white;
    border-radius: 8px;
    padding: 8px;
    border: 1px solid #555;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title('Stock Market Prediction')

# ----------------------------
# Robust yfinance download
# ----------------------------
@st.cache_data(ttl=60 * 10, show_spinner=False)
def yf_download_safe(ticker: str, start=None, end=None, period=None, interval="1d"):
    """
    Safer wrapper:
    - threads=False (important on Streamlit Cloud)
    - retries with small backoff
    - normalize MultiIndex columns
    """
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")

            df = df.reset_index()

            # Some returns have MultiIndex columns like ('Close','BBCA.JK')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Ensure Date exists
            if "Date" not in df.columns and "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})

            return df

        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt * 1.5)  # backoff

    raise RuntimeError(f"yfinance failed after retries: {last_err}")

# ----------------------------
# Inputs
# ----------------------------
ticker = st.sidebar.text_input('Code Saham', 'BBCA.JK')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

period_options = {
    '1 Day': '1d',
    '5 Days': '5d',
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '5 Years': '5y'
}
selected_period = st.sidebar.selectbox("Pilih Rentang Waktu", list(period_options.keys()))
prediction_days_num = int(st.sidebar.text_input('Prediksi N Hari Kedepan', 1))

years = (end_date - start_date).days // 365
if years < 5:
    st.warning("Untuk hasil yang lebih baik, rentang waktu harus minimal 5 tahun")
    st.stop()

# ----------------------------
# Download + show data
# ----------------------------
try:
    data = yf_download_safe(ticker, start=start_date, end=end_date, interval="1d")
except Exception as e:
    st.error("Gagal download data dari Yahoo Finance (yfinance). Ini bisa terjadi kalau Yahoo memblokir server Streamlit Cloud.")
    st.exception(e)
    st.info("Coba: ganti ticker (misal AAPL), coba lagi beberapa menit, atau pakai Twelve Data untuk stabil.")
    st.stop()

st.subheader(f'Stock Data From {start_date} To {end_date}')
st.write(data)

# ----------------------------
# MA charts
# ----------------------------
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], ma100, label='100-Day Moving Average')
plt.title(f'{ticker} Closing Price and 100-Day MA')
plt.legend(loc='best')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma200 = data['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], ma100, label='100-Day Moving Average')
plt.plot(data['Date'], ma200, label='200-Day Moving Average')
plt.title(f'{ticker} Closing Price, 100-Day and 200-Day MAs')
plt.legend(loc='best')
st.pyplot(fig)

# ----------------------------
# Candlestick
# ----------------------------
st.subheader(f'Candlestick Chart ({selected_period})')
try:
    data_candle = yf_download_safe(ticker, period=period_options[selected_period], interval="1d")
except Exception as e:
    st.error("Gagal ambil candlestick data dari yfinance.")
    st.exception(e)
    st.stop()

fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=data_candle['Date'],
            open=data_candle['Open'],
            high=data_candle['High'],
            low=data_candle['Low'],
            close=data_candle['Close'],
            name='Candlestick'
        )
    ]
)
fig_candle.update_layout(
    title=f'{ticker} Candlestick Chart ({selected_period})',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig_candle, use_container_width=True)

# ----------------------------
# Train/Test split
# ----------------------------
train_data = pd.DataFrame(data['Close'][0:int(len(data) * 0.7)])
test_data  = pd.DataFrame(data['Close'][int(len(data) * 0.7):])

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_array = scaler.fit_transform(train_data)

x_train, y_train = [], []
for i in range(100, train_data_array.shape[0]):
    x_train.append(train_data_array[i - 100:i])
    y_train.append(train_data_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# ----------------------------
# Load model once
# ----------------------------
@st.cache_resource
def load_cached_model(path: str):
    return load_model(path)

model = load_cached_model('adds/Final_Model.h5')

# ----------------------------
# Test prep
# ----------------------------
past_100_days = train_data.tail(100)
final_df = pd.concat([past_100_days, test_data], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()

# ----------------------------
# Prediction vs Original
# ----------------------------
st.subheader('Prediction VS Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Original Price')
plt.plot(y_pred, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# ----------------------------
# Forward N-day prediction
# ----------------------------
st.subheader('Prediction Result : ')
try:
    data2 = yf_download_safe(ticker, period='100d', interval='1d')
except Exception as e:
    st.error("Gagal ambil 100d data untuk prediksi N hari.")
    st.exception(e)
    st.stop()

# Keep only Close
train_data2 = pd.DataFrame(data2['Close'])
scaler2 = MinMaxScaler(feature_range=(0, 1))
train_data2_scaled = scaler2.fit_transform(train_data2).reshape(1, 100, 1)

n = prediction_days_num
temp_pred = []

for i in range(n):
    y_next = model.predict(train_data2_scaled[:, i:train_data2_scaled.shape[1], :], verbose=0)
    train_data2_scaled = np.concatenate((train_data2_scaled, y_next.reshape(1, 1, 1)), axis=1)
    temp_pred.append(scaler2.inverse_transform(y_next).reshape(-1)[0])

temp_pred2 = pd.DataFrame(temp_pred, columns=['Predicted Price'])
temp_pred2.index = range(0, n)
temp_pred2.index.name = 'Hari'
st.write(temp_pred2)

# Plot last 7d + prediction
try:
    temp_data1 = yf_download_safe(ticker, period='7d', interval='1d')
except Exception as e:
    st.error("Gagal ambil 7d data.")
    st.exception(e)
    st.stop()

temp_close = temp_data1['Close'].to_numpy().reshape(-1, 1)
pred_arr = np.array(temp_pred).reshape(-1, 1)
fusion = np.vstack((temp_close, pred_arr))

st.subheader('Stock Price with Predictions')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(fusion, label='Stock Price & Prediction')
ax.axvline(x=len(temp_close), linestyle='dashed', label='Prediction Start')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# ----------------------------
# DP max profit
# ----------------------------
def max_profit_dp(prices: list, k: int):
    n = len(prices)
    dp = {}

    def f(i, stats, k):
        if i >= n or k < 0:
            return [0, ""]
        if (i, stats, k) in dp:
            return dp[(i, stats, k)]

        if stats == "buy":
            a, path1 = f(i + 1, "sell", k - 1)
            a -= prices[i]
            b, path2 = f(i + 1, "buy", k)
            dp[(i, stats, k)] = [a, f"buy at {i} -> {path1}"] if a > b else [b, path2]
        else:
            a, path1 = f(i + 1, "buy", k)
            a += prices[i]
            b, path2 = f(i + 1, "sell", k)
            dp[(i, stats, k)] = [a, f"sell at {i} -> {path1}"] if a > b else [b, path2]

        return dp[(i, stats, k)]

    result = f(0, "buy", k)
    return result[0], result[1]

predicted_prices = temp_pred2['Predicted Price'].tolist()
profit, strategy_path = max_profit_dp(predicted_prices, 100)

st.subheader("Maximal Profit Strategy")
st.markdown(
    f"""
    <div style='background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; color: white'>
        <p><strong>Total Max Profit:</strong> {profit:.2f}</p>
        <p><strong>Strategi:</strong> {strategy_path}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Webhook (as-is)
# ----------------------------
import requests
payload = {
    "ticker": ticker,
    "start_date": str(start_date),
    "end_date": str(end_date),
    "predicted_prices": temp_pred2['Predicted Price'].tolist()
}
webhook_url = "https://nominally-picked-grubworm.ngrok-free.app/webhook/stockanalysis"

st.subheader("Stock Market Analysis")
try:
    response = requests.post(webhook_url, json=payload, timeout=30)
    if response.status_code == 200:
        st.success("Data berhasil dikirim!")
        import base64
        from io import BytesIO
        from PIL import Image

        base64_str = response.json()['base64StringImage'][0]
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption="Analisis Chart", use_container_width=True)

        analysis_text = response.json()['content'][0]
        st.markdown(
            f"""
            <div style='background-color: rgba(0, 0, 0, 0.5); color: white; padding: 20px; border-radius: 10px; font-family: Arial, sans-serif; font-size: 16px;'>
                {analysis_text}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Gagal kirim data. Status: {response.status_code}")
except Exception as e:
    st.error(f"Terjadi error webhook: {e}")

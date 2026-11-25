
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from pathlib import Path

st.set_page_config(
    page_title="Prediksi Penjualan MI-006",
    layout="wide"
)

st.title("Prediksi Penjualan Harian Produk Susu MI-006")

model = tf.keras.models.load_model(
    "model/lstm_mi006.keras"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

hasil = pd.read_csv(
    "model/hasil_prediksi.csv"
)

hasil['Tanggal'] = pd.to_datetime(
    hasil['Tanggal']
)

st.subheader("Ringkasan Evaluasi Model")

c1, c2, c3 = st.columns(3)

# Isi metrik dengan nilai evaluasi yang sudah dihitung
c1.metric("MAE", f"{hasil['Aktual'].sub(hasil['Prediksi']).abs().mean():.2f}")

rmse = np.sqrt(
    np.mean(
        (hasil['Aktual'] - hasil['Prediksi'])**2
    )
)

c2.metric("RMSE", f"{rmse:.2f}")

mape = np.mean(
    np.abs(
        (hasil['Aktual'] - hasil['Prediksi']) /
        hasil['Aktual'].replace(0, np.nan)
    )
) * 100

c3.metric("MAPE", f"{mape:.2f}%")

st.subheader("Grafik Aktual vs Prediksi")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    hasil['Tanggal'],
    hasil['Aktual'],
    label='Aktual'
)

ax.plot(
    hasil['Tanggal'],
    hasil['Prediksi'],
    label='Prediksi LSTM'
)

ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Penjualan")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.subheader("Prediksi Penjualan Hari Berikutnya")

last_sequence = np.load(
    "model/last_sequence.npy"
).reshape(1, 30, 1)

pred_scaled = model.predict(
    last_sequence,
    verbose=0
)

prediksi = scaler.inverse_transform(
    pred_scaled
)[0][0]

st.metric(
    "Estimasi Penjualan",
    f"{max(0, prediksi):.0f} unit"
)

st.caption(
    "Hasil prediksi merupakan estimasi berdasarkan "
    "30 hari data historis terakhir."
)

st.subheader("Data Hasil Prediksi")

st.dataframe(
    hasil,
    use_container_width=True
)

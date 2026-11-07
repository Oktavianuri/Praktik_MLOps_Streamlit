import streamlit as st
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Kelulusan", page_icon="🎓")
st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")
st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi kelulusan.")

# 2. Load Model (Arsitektur: Memisahkan UI dari Model)
@st.cache_resource
def load_model():
    return joblib.load('model_kelulusan.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("File model tidak ditemukan! Pastikan file 'model_kelulusan.pkl' sudah dipindahkan ke folder ini.")
    st.stop()

# 3. Membuat Form Input (UI)
with st.form("form_prediksi"):
    st.subheader("Masukkan Data Mahasiswa")
    ipk = st.number_input("IPK (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    sks = st.number_input("Total SKS", min_value=0, max_value=200, value=144, step=1)
    tombol = st.form_submit_button("Prediksi Kelulusan")

# 4. Logika ketika tombol ditekan
if tombol:
    # Prediksi menggunakan model
    hasil = model.predict([[ipk, sks]])
    
    # Tampilkan Output
    st.divider()
    st.subheader("Hasil Prediksi")
    if hasil[0] == 1:
        st.success(f"Mahasiswa dengan IPK **{ipk}** dan SKS **{sks}** diprediksi **LULUS TEPAT WAKTU**.")
    else:
        st.error(f"Mahasiswa dengan IPK **{ipk}** dan SKS **{sks}** diprediksi **TIDAK LULUS**.")
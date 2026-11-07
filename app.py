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
    st.error("File model tidak ditemukan! Jalankan `1_latih_model.py` terlebih dahulu.")
    st.stop()

# 3. Membuat Form Input (UI)
with st.form("form_prediksi"):
    st.subheader("Masukkan Data Mahasiswa")
    st.caption("Syarat Kelulusan: Minimal IPK 2.00 dan Total SKS wajib 144 SKS.")
    ipk = st.number_input("IPK (0.00 - 4.00)", min_value=0.0, value=3.0, step=0.01)
    sks = st.number_input("Total SKS (Maksimal 144)", min_value=0, value=144, step=1)
    tombol = st.form_submit_button("Prediksi Kelulusan")

# 4. Logika ketika tombol ditekan
if tombol:
    st.divider()
    st.subheader("Hasil Prediksi")

    # Validasi batas atas: Tidak bisa dicari jika melebihi IPK 4.00 atau 144 SKS
    if ipk > 4.00 or sks > 144:
        st.warning("⚠️ **Data Tidak Dapat Dicari!** Nilai input melebihi batas maksimal (IPK maksimal **4.00** dan Total SKS maksimal **144 SKS**).")
    else:
        # Prediksi menggunakan model jika input valid
        hasil = model.predict([[ipk, sks]])

        if hasil[0] == 1:
            st.success(f"Mahasiswa dengan IPK **{ipk:.2f}** dan SKS **{sks}** diprediksi **LULUS TEPAT WAKTU**.")
        else:
            st.error(f"Mahasiswa dengan IPK **{ipk:.2f}** dan SKS **{sks}** diprediksi **TIDAK LULUS** (Belum memenuhi syarat minimal IPK 2.00 dan/atau 144 SKS).")
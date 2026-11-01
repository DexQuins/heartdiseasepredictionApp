import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Awal ---
st.set_page_config(
    page_title="Prediksi Kerentanan Penyakit Jantung",
    layout="wide"
)

# --- 2. Muat Model dan Fitur ---
try:
    model = joblib.load('heart_disease_model.pkl')
    # Muat nama fitur yang digunakan saat training (sangat penting untuk prediksi)
    feature_names = joblib.load('feature_names.pkl')
except FileNotFoundError:
    st.error("File model 'heart_disease_model.pkl' atau 'feature_names.pkl' tidak ditemukan. Pastikan Anda telah menjalankan 'model_training.py' terlebih dahulu.")
    st.stop()
    
# --- 3. Desain UI Aplikasi ---
st.title("❤️ Aplikasi Prediksi Kerentanan Penyakit Jantung")
st.markdown("Aplikasi ini menggunakan Model Klasifikasi Machine Learning untuk memprediksi risiko penyakit jantung berdasarkan input kesehatan.")

# --- Sidebar untuk Input Data ---
st.sidebar.header("Input Data Pasien")

# Fungsi untuk mendapatkan input dari user
def get_user_input():
    # Sesuaikan batas dan opsi berdasarkan dataset Anda
    age = st.sidebar.slider('Usia (Tahun)', 20, 80, 50)
    sex = st.sidebar.selectbox('Jenis Kelamin', options=[(0, 'Perempuan'), (1, 'Laki-laki')], format_func=lambda x: x[1])[0]
    
    # Chest Pain Type (cp)
    cp_options = {0: 'Asymptomatic', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Typical Angina'}
    cp = st.sidebar.selectbox('Tipe Nyeri Dada (cp)', options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

    trestbps = st.sidebar.number_input('Tekanan Darah Istirahat (mmHg)', 90, 200, 120)
    chol = st.sidebar.number_input('Kolesterol Serum (mg/dl)', 100, 600, 250)
    
    # Fasting Blood Sugar > 120 mg/dl (fbs)
    fbs = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl', options=[(0, 'Tidak'), (1, 'Ya')], format_func=lambda x: x[1])[0]

    thalach = st.sidebar.number_input('Detak Jantung Maksimal (thalach)', 70, 210, 150)
    exang = st.sidebar.selectbox('Angina yang Diinduksi Olahraga', options=[(0, 'Tidak'), (1, 'Ya')], format_func=lambda x: x[1])[0]
    oldpeak = st.sidebar.number_input('Depresi ST yang Diinduksi Olahraga', 0.0, 6.2, 1.0)
    
    # Slope of the peak exercise ST segment (slope)
    slope_options = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
    slope = st.sidebar.selectbox('Slope ST Segment Puncak Latihan', options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

    ca = st.sidebar.slider('Jumlah Pembuluh Darah Utama (ca)', 0, 4, 0)
    
    # Thal (thallium stress result)
    thal_options = {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Lainnya'}
    thal = st.sidebar.selectbox('Thal Result', options=list(thal_options.keys()), format_func=lambda x: thal_options[x])
    
    # restecg (Resting Electrocardiographic Results)
    restecg_options = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
    restecg = st.sidebar.selectbox('Hasil EKG Istirahat', options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])


    # Data harus dikembalikan sebagai dictionary agar mudah dibuat DataFrame
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(user_data, index=[0])

# Dapatkan input user
input_df = get_user_input()

# --- 4. Tampilkan Input dan Prediksi ---

st.subheader("Data Input Pasien:")
st.dataframe(input_df)

if st.button("Prediksi Risiko Penyakit Jantung"):
    # Karena model pipeline sudah menangani pre-processing (scaling dan one-hot encoding),
    # kita hanya perlu memastikan urutan kolom dan namanya sama saat predict.
    
    try:
        # Gunakan model pipeline untuk prediksi
        prediction = model.predict(input_df)
        
        # Jika model adalah Regresi Logistik atau sejenis, kita bisa mendapatkan probabilitas
        prediction_proba = model.predict_proba(input_df)
        risk_percentage = prediction_proba[0][1] * 100 # Probabilitas kelas 1 (berisiko)

        st.subheader("Hasil Prediksi")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **Risiko Penyakit Jantung:** **Tinggi**")
            st.markdown(f"**Tingkat Probabilitas Risiko:** **{risk_percentage:.2f}%**")
            st.warning("Hasil ini mengindikasikan bahwa model memprediksi risiko penyakit jantung tinggi. Mohon **segera konsultasi** dengan profesional kesehatan untuk diagnosis dan tindak lanjut.")
        else:
            st.success(f"✅ **Risiko Penyakit Jantung:** **Rendah**")
            st.markdown(f"**Tingkat Probabilitas Risiko:** **{risk_percentage:.2f}%**")
            st.info("Risiko rendah yang diprediksi model. Namun, terus pertahankan gaya hidup sehat dan lakukan pemeriksaan rutin.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
        st.info("Pastikan kolom-kolom pada input data sesuai dengan data yang digunakan saat training.")

st.markdown("---")
st.caption("Disclaimer: Aplikasi ini adalah alat bantu prediksi berbasis ML dan bukan pengganti diagnosis medis profesional.")
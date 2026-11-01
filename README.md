# 🫀 Heart Disease Classification & Prediction App

Aplikasi web berbasis Machine Learning untuk memprediksi potensi risiko penyakit jantung pada pasien berdasarkan indikator klinis medis.

---

## 📌 Deskripsi Proyek
Proyek ini memanfaatkan model Machine Learning yang dilatih menggunakan dataset kesehatan (`heart_disease_data.csv`) untuk melakukan evaluasi dan prediksi risiko dini penyakit jantung secara cepat dan akurat melalui antarmuka web Streamlit.

---

## 🛠️ Teknologi & Tool
* **Bahasa Pemrograman:** Python
* **Machine Learning & Processing:** Scikit-Learn, Pandas, NumPy, Joblib
* **Web Framework:** Streamlit

---

## 📁 Struktur File Proyek
```text
streamlit/
├── app.py                     # Aplikasi UI utama (Streamlit)
├── model_training.py          # Script pelatihan data & evaluasi model
├── heart_disease_data.csv     # Dataset indikator medis pasien
├── heart_disease_model.pkl    # Model ML yang sudah dilatih (Trained Model)
├── feature_names.pkl          # Daftar nama fitur pendukung model
└── README.md                  # Dokumentasi proyek

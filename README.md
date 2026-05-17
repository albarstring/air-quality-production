# 🌫️ AirSense – Dashboard Prediksi Kualitas Udara

Dashboard Streamlit modern berbasis **LightGBM** untuk memprediksi kualitas udara (AQI)
menggunakan data polutan dan cuaca secara realtime.

## 📦 Struktur Proyek

```
airquality_app/
├── app.py               ← Aplikasi Streamlit utama
├── train_model.py       ← Script training model (jalankan sekali)
├── requirements.txt     ← Dependensi Python
└── model_output/        ← Dibuat otomatis setelah training
    ├── LightGBM.pkl
    ├── scaler.pkl
    └── metadata.json
```

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Latih model (hanya sekali)
```bash
python train_model.py
```

### 3. Jalankan dashboard
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

## 🎯 Fitur Dashboard

| Fitur | Deskripsi |
|-------|-----------|
| **Prediksi AQI realtime** | Prediksi otomatis setiap slider bergerak |
| **Gauge AQI** | Visualisasi jarum meter berwarna |
| **Prakiraan 24 Jam** | Bar chart proyeksi AQI sepanjang hari |
| **Analisis Sensitivitas** | Grafik pengaruh PM2.5 & angin terhadap AQI |
| **Cuaca Realtime** | Data langsung dari Open-Meteo API |
| **Early Warning** | Alert berwarna + rekomendasi tindakan |
| **Referensi AQI** | Tabel standar kategori US EPA |

## 🤖 Spesifikasi Model

- **Algoritma**: LightGBM (Light Gradient Boosting Machine)
- **Fitur**: 19 fitur (polutan + cuaca + feature engineering)
- **Performa**: MAE ≈ 3.4 | RMSE ≈ 4.3 | R² ≈ 0.965

## 🎨 Desain

Dark industrial dashboard dengan palet biru-ungu, tipografi Syne + JetBrains Mono,
dan visualisasi Matplotlib bertema gelap.

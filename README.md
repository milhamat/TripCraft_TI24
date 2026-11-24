# TripCraft Batam

Sistem rekomendasi wisata dan penyusun *itinerary* otomatis untuk Kota Batam, dibangun dengan **Content-Based Filtering** (Term Frequency–Inverse Document Frequency + cosine similarity) dan dideploy sebagai **aplikasi web statis tanpa backend**.

Proyek UAS Pembelajaran Mesin — Program Studi Teknologi Informasi, Universitas Internasional Batam.

**Tim:** Hendra Cung, Shinta Natalia, Xaverio, Tasya Pardede, Firman

## Fitur

- Pencarian destinasi dengan bahasa sehari-hari ("tempat ngopi di nagoya", "pantai untuk keluarga")
- 154 POI Kota Batam terverifikasi (koordinat & rating via Google Places) pada 6 kategori: Kuliner, Hotel, Wisata, Religi, Belanja, Alam
- Badge **Recommended** berbasis skor kualitas deterministik (0,6×norm(rating) + 0,4×norm(log(1+ulasan)))
- Penyusun itinerary otomatis multi-hari: jarak Haversine, kawasan jangkar per hari, slot waktu, prioritas Recommended, keberagaman kategori
- Berjalan langsung dari satu file HTML — tanpa server, tanpa instalasi

## Arsitektur

| Komponen | Teknik | Berjalan di website? |
|---|---|---|
| Retrieval pencarian | TF-IDF (1–2 gram, sublinear, L2) + cosine similarity | ✅ Ya (replikasi JS identik dengan scikit-learn) |
| Klasifikasi kategori | KNN (k=5, cosine) vs Naive Bayes vs Decision Tree vs Logistic Regression | ❌ Tidak — komponen evaluasi saja |
| Badge Recommended | Skor kualitas deterministik (bukan ML) | ✅ Ya |
| Itinerary | Heuristik (Haversine + slot waktu + variasi kategori) | ✅ Ya |

Hasil evaluasi klasifikasi kategori: **KNN 94,87%** akurasi test / **93,51%** cross-validation 5-fold, mengungguli Decision Tree, Naive Bayes, dan Logistic Regression (masing-masing 89,74%).

## Struktur Repositori

```
├── 01_eksplorasi_data_batam.ipynb   # EDA + pembersihan data
├── 02_model_cbf_knn_batam.ipynb     # Training & evaluasi model
├── 03_itinerary_batam.ipynb         # Logika penyusun itinerary
├── build_recommend.py               # Ekspor model TF-IDF ke aplikasi HTML
├── tripcraft_batam.html             # Aplikasi web (buka langsung di browser)
├── data/
│   ├── cleaned/batam_cleaned.csv    # Dataset utama (154 POI)
│   ├── fasilitas_hotel.json
│   └── fasilitas_wisata.json
└── models/                          # Artefak model (joblib)
```

## Cara Menjalankan

1. Jalankan `01_eksplorasi_data_batam.ipynb` (menghasilkan `data/cleaned/batam_cleaned.csv`)
2. `python build_recommend.py` (menyuntik model TF-IDF ke `tripcraft_batam.html`)
3. Jalankan `02_model_cbf_knn_batam.ipynb` dan `03_itinerary_batam.ipynb` untuk evaluasi
4. Buka `tripcraft_batam.html` di browser (butuh internet untuk peta)

**Dependensi:** Python 3, scikit-learn, pandas, matplotlib

## Keterbatasan yang Diketahui

Retrieval bersifat leksikal — kueri dengan kosakata di luar korpus (mis. "snorkeling", "vegetarian") tidak mengembalikan hasil. Lihat bagian Saran pada artikel untuk arah pengembangan (embedding semantik, hybrid filtering).

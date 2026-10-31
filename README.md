# Dashboard Rekomendasi Buku (Book-Crossing)

Struktur:
```
bookrec/
├── data/            <- taruh BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv
├── artifacts/       <- hasil training (otomatis)
├── train_model.py   <- preprocessing + training + evaluasi
├── recsys.py        <- logika rekomendasi (inferensi)
├── app.py           <- dashboard Streamlit
└── requirements.txt
```
Langkah:
```
pip install -r requirements.txt
python train_model.py        # ± 1 menit (hybrid CF + Content-Based), membuat artifacts/model.joblib
streamlit run app.py
```

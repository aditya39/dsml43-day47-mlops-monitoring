# DSML 43 — Day 47: Python-Powered MLOps (Hands-on)

Repo hands-on untuk sesi **Building and Monitoring ML Pipelines**. Kalian mengubah API harga rumah Day 46
menjadi sistem MLOps kecil: MLflow (tracking + registry), FastAPI yang memuat model `@champion`,
promote dan rollback, log JSON, metrik Prometheus, dashboard Grafana, dan laporan drift.

## Mulai dari sini

Buka **[`lab/README.md`](lab/README.md)** dan ikuti langkahnya (Persiapan, lalu Bagian 0 sampai 3).
Kerjakan di folder `lab/starter`.

📘 **Buku panduan peserta (PDF):** [`docs/Buku-Panduan-Peserta-Day-47.pdf`](docs/Buku-Panduan-Peserta-Day-47.pdf), versi lengkap langkah demi langkah dengan gambar dan lembar kerja.

## Prasyarat (siapkan SEBELUM kelas)

- **Docker Desktop** terpasang dan berjalan (Windows: backend WSL2)
- **Python 3.11** dan **VS Code**
- Ruang disk kosong **±5 GB** (image Docker ±4 GB)
- Tarik image dan build API sebelum kelas, jangan saat kelas (download ±1,5 GB):
  ```
  cd lab/starter
  docker compose pull
  docker compose build
  ```

## Peta folder

```
.
├── README.md              <- file ini
├── assignment.md          <- tugas setelah sesi
└── lab/
    ├── README.md          <- panduan langkah demi langkah (mulai di sini)
    ├── starter/           <- tempat kalian bekerja (TODO 1-9)
    └── solution/          <- jawaban lengkap; pakai sebagai cadangan kalau mentok
```

## Tugas

Setelah sesi, kerjakan **[`assignment.md`](assignment.md)**.

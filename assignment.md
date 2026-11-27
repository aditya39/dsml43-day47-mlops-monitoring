# Assignment Day 47 — Deployment MLOps with Docker

**Individu · dikumpulkan di LMS · maksimal H+12 setelah kelas**
Kelas: Sabtu, 5 Desember 2026 → **batas: Kamis, 17 Desember 2026** (jam mengikuti yang tertera di LMS; kalau LMS berbeda,
ikuti LMS). Panduan langkah demi langkah: `lab/README.md` dan Buku Panduan Peserta (`docs/`)

---

## 1. Brief

Kamu adalah ML engineer di sebuah startup. Tim data science menyerahkan model yang "sudah jadi di notebook". Tugasmu
menjadikannya **produk yang bisa dipertanggungjawabkan**: bisa dilatih ulang dengan satu perintah, menolak model yang
jelek, punya versi dan jalan pulang (rollback), dilayani lewat API di Docker, dan dipantau setelah live.

Bangun ulang sistem dari lab Day 47 untuk **dataset tabular pilihanmu sendiri** (bukan dataset harga rumah sintetis lab).
Contoh: Kaggle *House Prices* (regresi), *Telco Customer Churn* (klasifikasi), *Credit Card Approval*, *Bike Sharing
Demand*, atau dataset dari proyekmu sendiri. Model boleh apa saja dari scikit-learn (atau XGBoost/LightGBM) selama
preprocessing + model disimpan sebagai **satu Pipeline**.

**Titik awal yang disarankan:** salin `lab/solution`, lalu ganti dataset, fitur, model, schema Pydantic, metrik quality
gate, dan panel drift. Kerangka boleh sama; isi dan keputusan harus milikmu.

---

## 2. Deliverables

Kumpulkan **link repo GitHub publik** (atau ZIP kalau repo privat) + **link video demo** di LMS.

### 2.1 Struktur repo minimal

```
README.md                  # cara menjalankan dari nol + screenshot + penjelasan keputusan
requirements.txt           # versi dikunci (==), satu file untuk training & serving
config.yaml                # angka keputusan (path, hyperparameter, batas gate, alamat MLflow)
.env.example  .gitignore   # .env, data mentah besar, mlruns/ TIDAK di-commit
src/                       # config.py, data.py (load+validasi+split), train.py, evaluate.py
app/                       # FastAPI: schemas.py, main.py (/health, /predict, /metrics, /admin/reload)
scripts/                   # promote.py, rollback.py, traffic.py, drift_report.py, smoke_test.py
monitoring/                # prometheus.yml (+ alerts.yml), provisioning & dashboard Grafana
Dockerfile                 # multi-stage, python:3.11-slim (atau versi yang kamu kunci)
docker-compose.yml         # minimal: mlflow + api + prometheus + grafana
```

### 2.2 Yang harus bisa dibuktikan (masukkan screenshot di README)

1. `python -m src.train` dua kali: satu **lolos** quality gate (terdaftar), satu **ditolak** (mis. model sengaja dangkal) —
   screenshot terminal + daftar run di MLflow (FINISHED dan FAILED).
2. MLflow Registry dengan **≥ 2 versi** dan alias `@champion` (+ `@challenger`/`@previous`).
3. **Promote** lalu **rollback**: output script + `/health` yang menunjukkan versi berubah.
4. API berjalan di Docker Compose, `python scripts/smoke_test.py` → lulus.
5. Contoh **2–3 baris log JSON** dari `docker compose logs api`.
6. **Dashboard Grafana** dengan minimal: request rate, p95 latency, error rate, versi model yang melayani, dan satu panel
   distribusi fitur/prediksi.
7. **Laporan drift**: `drift_report.py` pada data "normal" (tidak drift) dan data yang kamu geser sendiri (drift) — tabel
   PSI + p-value per fitur dan exit code. Jelaskan cara kamu membuat data drift (mis. subset waktu terbaru, filter segmen,
   atau perturbasi yang masuk akal).

### 2.3 Video demo (maks 5 menit)

Tunjukkan: `docker compose up` → train (lolos & ditolak) → MLflow registry → promote → API melayani versi baru → rollback
→ traffic → Grafana → drift report. Suara/teks penjelasan singkat. Tidak perlu edit rapi; yang penting jalan.

### 2.4 README minimal berisi

- Tujuan model & dataset (sumber + lisensi), metrik yang dipakai dan **kenapa** batas quality gate-nya segitu.
- Langkah menjalankan dari nol (Windows dan/atau macOS) — orang lain harus bisa mengikutinya.
- Arsitektur (diagram teks/mermaid) dan daftar port.
- Screenshot bukti (§2.2).
- Satu paragraf "apa yang akan saya lakukan kalau ini produksi sungguhan" (mis. auth untuk endpoint admin, Alertmanager,
  jadwal retraining, data versioning).

---

## 3. Rubrik penilaian (total 100)

| # | Aspek | Poin | Penuh kalau… | Sebagian kalau… |
|---|---|---|---|---|
| 1 | **Pipeline modular + validasi data** | 15 | Logika di `src/` (bukan notebook); config di `config.yaml`/env; validasi data minimal 3 cek yang relevan dengan dataset; seed tetap; `python -m src.train` jalan dari nol | Modul ada tapi masih ada hardcode/path absolut, atau validasi minim |
| 2 | **MLflow: tracking, quality gate, registry & rollback** | 30 | Params, metrics, tags (git_sha/data hash) tercatat; gate pakai `raise` dan model yang gagal **tidak** terdaftar; ≥ 2 versi; alias `@champion`; promote membandingkan metrik di test set yang sama; rollback memindahkan alias dan terbukti di `/health` | Tracking jalan tapi gate hanya print, atau tidak ada bukti rollback |
| 3 | **FastAPI + Docker Compose** | 20 | API memuat model dari registry via alias (bukan file di image); validasi Pydantic + error handling (422/503); compose menjalankan mlflow + api + prometheus + grafana; smoke test lulus | API masih memuat file `.pkl` dari image, atau compose hanya sebagian |
| 4 | **Logging & monitoring** | 20 | Satu baris log JSON per request (request_id, model_version, fitur, prediksi, latency; tanpa data pribadi); `/metrics` dengan counter + histogram; dashboard Grafana ter-provision dari file (minimal 5 panel di §2.2 no. 6) | Log masih teks bebas, atau dashboard dibuat manual tanpa file provisioning |
| 5 | **Drift report + dokumentasi** | 15 | Drift report (PSI + KS/chi²) pada data normal vs drift dengan interpretasi; README bisa diikuti orang lain; video ≤ 5 menit | Drift report tanpa interpretasi, README kurang langkah, atau video tidak ada |
| | **Total** | **100** | | |

**Pengurangan nilai:**
- Rahasia (password, token, API key) ter-commit di repo atau notebook: **−15**.
- `.env`, `mlruns/`, atau file data besar (> 50 MB) masuk Git: **−5**.
- Repo tidak bisa dijalankan sama sekali dengan langkah di README: nilai maksimal aspek 3–5 dipotong setengah.

**Bonus (maks +10, total nilai tetap maksimal 100):** alert Prometheus yang bermakna (+3) · endpoint admin dilindungi token
dari env (+3) · retraining terjadwal (cron/Prefect) yang memanggil pipeline yang sama (+4).

---

## 4. Aturan & etika

- **Individu.** Diskusi boleh; kode dan README harus tulisanmu sendiri.
- Penggunaan AI mengikuti aturan dibimbing yang berlaku. Apa pun alat bantunya, kamu harus bisa **menjelaskan setiap baris**
  yang kamu kumpulkan — mentor dapat meminta penjelasan singkat.
- Hormati lisensi dataset; sebutkan sumbernya. Jangan memakai data pribadi tanpa izin; kalau dataset berisi kolom identitas
  (nama, NIK, email, nomor HP), buang atau hash sebelum dipakai, dan **jangan pernah** mencatatnya di log.
- Keterlambatan mengikuti aturan LMS/kelas.

---

## 5. Checklist sebelum submit

- [ ] Clone repo ke folder baru → ikuti README dari nol → semuanya jalan.
- [ ] `docker compose up -d --build` → 4 service Up; `python scripts/smoke_test.py` → lulus.
- [ ] Ada run FINISHED **dan** FAILED di MLflow; registry ≥ 2 versi; alias benar.
- [ ] Screenshot Grafana menampilkan data (bukan "No data").
- [ ] `git status` bersih dari `.env`, `mlruns/`, data besar; `git log` menunjukkan beberapa commit bertahap.
- [ ] Link repo + link video sudah dicek bisa dibuka dari akun lain (mode incognito).

## 6. Tips

- Mulai dari yang jalan: salin `lab/solution`, jalankan apa adanya, **baru** ganti dataset sedikit demi sedikit (data →
  schema → model → gate → dashboard). Commit setiap langkah yang berhasil.
- Klasifikasi? Ganti MAE dengan F1/ROC-AUC dan balik arah gate-nya (`if f1 < min_f1: raise ...`). Histogram prediksi bisa
  diganti counter per kelas prediksi.
- Pilih batas quality gate dari baseline sederhana (mis. model rata-rata / dummy classifier) — jelaskan alasannya di README.
- Pakai `--allowed-hosts` di server MLflow 3.x kalau API memanggilnya lewat nama service compose (lihat Troubleshooting di `lab/README.md`).
- Simpan screenshot sambil mengerjakan; jangan menunggu malam deadline.

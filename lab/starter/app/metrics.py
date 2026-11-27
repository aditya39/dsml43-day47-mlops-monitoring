"""app/metrics.py — metrik Prometheus. /metrics cuma teks; Prometheus yang datang membacanya (pull)."""
from prometheus_client import Counter, Gauge, Histogram

# COUNTER: hanya naik. Dipakai sebagai laju (request/detik) dan error rate.
REQUESTS = Counter(
    "api_requests_total", "Jumlah request HTTP", ["endpoint", "status"]
)
PREDICTIONS = Counter(
    "predictions_total", "Jumlah prediksi sukses", ["model_version"]
)

# HISTOGRAM: sebaran nilai -> persentil (p50, p95), bukan cuma rata-rata.
LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency request HTTP (detik)",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTED_PRICE = Histogram(
    "predicted_price_rupiah",
    "Sebaran harga hasil prediksi (Rupiah)",
    buckets=(5e8, 1e9, 1.5e9, 2e9, 3e9, 5e9, 8e9, 12e9),
)
FEATURE_JARAK = Histogram(
    "input_jarak_ke_pusat_km",
    "Sebaran fitur jarak_ke_pusat_km di request (untuk memantau drift)",
    buckets=(5, 10, 15, 20, 25, 30, 40, 50, 70),
)

# GAUGE: nilai saat ini, bisa naik-turun. Di sini: versi model yang sedang melayani (=1).
MODEL_INFO = Gauge(
    "model_version_info", "Versi model yang sedang melayani (nilai selalu 1)", ["version"]
)

# Buat seri /predict sejak awal (nilai 0) supaya grafik error rate menunjukkan 0, bukan "No data".
for _status in ("200", "422", "500", "503"):
    REQUESTS.labels(endpoint="/predict", status=_status)

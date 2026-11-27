"""scripts/smoke_test.py — cek cepat API sehat. Exit code 1 kalau ada yang gagal.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://localhost:8000 --expect-version 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

SAMPLE = {
    "luas_tanah_m2": 120,
    "luas_bangunan_m2": 150,
    "kamar_tidur": 3,
    "kamar_mandi": 2,
    "jarak_ke_pusat_km": 12.5,
    "umur_bangunan_tahun": 5,
    "kota": "Depok",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--expect-version", help="versi model yang seharusnya dilayani")
    ap.add_argument("--wait", type=int, default=30, help="tunggu API siap (detik)")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    failures = []

    def check(name: str, ok: bool, info: str = "") -> None:
        print(f"[{'OK' if ok else 'GAGAL'}] {name} {info}")
        if not ok:
            failures.append(name)

    deadline = time.time() + args.wait
    while True:
        try:
            health = requests.get(f"{url}/health", timeout=5)
            break
        except requests.RequestException:
            if time.time() > deadline:
                print(f"[GAGAL] API tidak bisa dihubungi di {url}")
                return 1
            time.sleep(2)

    body = health.json()
    check("GET /health 200", health.status_code == 200)
    check("model sudah dimuat", body.get("model_version") is not None,
          f"(version={body.get('model_version')}, detail={body.get('detail')})")
    if args.expect_version:
        check(f"melayani v{args.expect_version}", str(body.get("model_version")) == str(args.expect_version))

    r = requests.post(f"{url}/predict", json=SAMPLE, timeout=10)
    ok = r.status_code == 200 and 1e8 < r.json().get("harga_prediksi", 0) < 5e10
    check("POST /predict valid -> 200 & harga masuk akal", ok, f"-> {r.status_code} {r.text[:120]}")

    bad = dict(SAMPLE, kota="Surabaya")
    r = requests.post(f"{url}/predict", json=bad, timeout=10)
    check("POST /predict kota tidak dikenal -> 422", r.status_code == 422, f"-> {r.status_code}")

    m = requests.get(f"{url}/metrics", timeout=10)
    check("GET /metrics berisi predictions_total", "predictions_total" in m.text)

    print("SMOKE TEST LULUS" if not failures else f"SMOKE TEST GAGAL: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

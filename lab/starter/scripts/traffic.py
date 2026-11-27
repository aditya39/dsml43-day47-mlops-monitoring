"""scripts/traffic.py — kirim trafik "produksi" ke API supaya dashboard Grafana hidup.

    python scripts/traffic.py --scenario live  --n 300       # data mirip training
    python scripts/traffic.py --scenario drift --n 600 --rate 5   # data bergeser (~2 menit)
    python scripts/traffic.py --scenario live --invalid 0.05  # + 5% request rusak (-> 422)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FILES = {"live": ROOT / "data" / "live_normal.csv", "drift": ROOT / "data" / "live_drift.csv"}
FEATURES = ["luas_tanah_m2", "luas_bangunan_m2", "kamar_tidur", "kamar_mandi",
            "jarak_ke_pusat_km", "umur_bangunan_tahun", "kota"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["live", "drift"], default="live")
    ap.add_argument("--n", type=int, default=300, help="jumlah request")
    ap.add_argument("--rate", type=float, default=20, help="request per detik")
    ap.add_argument("--invalid", type=float, default=0.0, help="proporsi request rusak (0-1)")
    ap.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    args = ap.parse_args()

    path = FILES[args.scenario]
    if not path.exists():
        print(f"[GAGAL] {path} tidak ada. Buat dulu:")
        print(f"        python make_dataset.py --scenario {args.scenario} --n 500 --out {path.relative_to(ROOT)}")
        return 1
    rows = pd.read_csv(path)[FEATURES].to_dict(orient="records")
    rng = random.Random(47)
    session = requests.Session()
    status, versions, latencies = Counter(), Counter(), []

    print(f"Mengirim {args.n} request ({args.scenario}) ke {args.url}/predict ...")
    for i in range(1, args.n + 1):
        payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in rng.choice(rows).items()}
        if rng.random() < args.invalid:
            payload["kota"] = "Surabaya"  # sengaja salah -> 422
        t0 = time.perf_counter()
        try:
            r = session.post(f"{args.url}/predict", json=payload, timeout=10)
            status[r.status_code] += 1
            if r.status_code == 200:
                versions[r.json()["model_version"]] += 1
        except requests.RequestException:
            status["koneksi gagal"] += 1
        latencies.append((time.perf_counter() - t0) * 1000)
        if i % 100 == 0:
            print(f"  {i}/{args.n} terkirim")
        time.sleep(max(0.0, 1 / args.rate - (time.perf_counter() - t0)))

    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    print(f"Selesai. status={dict(status)} versi_model={dict(versions)} p95_client={p95:.1f} ms")
    return 0 if status.get(200, 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

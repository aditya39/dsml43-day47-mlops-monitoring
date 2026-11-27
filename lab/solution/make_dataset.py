"""
make_dataset.py — Dataset sintetis "Harga Rumah Jabodetabek" untuk Day 46–48.

Dataset ini SINTETIS (dibuat dengan rumus + noise), deterministik (seed tetap),
dan hanya butuh numpy + pandas. Dipakai sebagai running example di tiga sesi:
  Day 46 → model disajikan via FastAPI + Docker
  Day 47 → pipeline + MLflow + monitoring (+ skenario drift)
  Day 48 → ditampilkan di portfolio Streamlit

Pemakaian:
  python make_dataset.py                          # data/train.csv (2000 baris, baseline)
  python make_dataset.py --scenario drift --n 500 --out data/live_drift.csv
  python make_dataset.py --scenario live  --n 500 --out data/live_normal.csv

Skenario:
  baseline : data training (seed 43)
  live     : data "produksi" yang distribusinya SAMA dengan training (seed beda)
  drift    : data "produksi" yang BERGESER —
             (1) data drift  : rumah makin jauh dari pusat kota & lebih banyak di Bogor/Bekasi
             (2) concept drift: harga naik ~18% (inflasi properti) untuk fitur yang sama
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOTA = [
    "Jakarta Selatan",
    "Jakarta Barat",
    "Jakarta Timur",
    "Depok",
    "Tangerang Selatan",
    "Bekasi",
    "Bogor",
]
# harga dasar per m2 tanah (Rupiah) & rentang jarak ke pusat kota (km)
HARGA_M2 = {
    "Jakarta Selatan": 22_000_000,
    "Jakarta Barat": 17_000_000,
    "Jakarta Timur": 14_000_000,
    "Depok": 9_000_000,
    "Tangerang Selatan": 12_000_000,
    "Bekasi": 8_500_000,
    "Bogor": 7_000_000,
}
JARAK_KM = {
    "Jakarta Selatan": (2, 15),
    "Jakarta Barat": (3, 18),
    "Jakarta Timur": (4, 20),
    "Depok": (15, 35),
    "Tangerang Selatan": (12, 32),
    "Bekasi": (14, 38),
    "Bogor": (25, 55),
}
P_KOTA_BASE = [0.18, 0.15, 0.15, 0.13, 0.15, 0.13, 0.11]
P_KOTA_DRIFT = [0.07, 0.08, 0.10, 0.17, 0.13, 0.22, 0.23]

FEATURES = [
    "luas_tanah_m2",
    "luas_bangunan_m2",
    "kamar_tidur",
    "kamar_mandi",
    "jarak_ke_pusat_km",
    "umur_bangunan_tahun",
    "kota",
]
TARGET = "harga"


def generate(n: int = 2000, scenario: str = "baseline", seed: int | None = None) -> pd.DataFrame:
    if scenario not in {"baseline", "live", "drift"}:
        raise ValueError("scenario harus baseline | live | drift")
    if seed is None:
        seed = {"baseline": 43, "live": 4301, "drift": 4302}[scenario]
    rng = np.random.default_rng(seed)

    p_kota = P_KOTA_DRIFT if scenario == "drift" else P_KOTA_BASE
    kota = rng.choice(KOTA, size=n, p=p_kota)

    luas_tanah = np.clip(rng.lognormal(mean=np.log(120), sigma=0.45, size=n), 36, 600).round()
    luas_bangunan = np.clip(luas_tanah * rng.uniform(0.6, 1.6, size=n), 30, 700).round()
    kamar_tidur = np.clip(np.round(luas_bangunan / 45 + rng.normal(0, 0.7, size=n)), 1, 7).astype(int)
    kamar_mandi = np.clip(np.round(kamar_tidur * rng.uniform(0.5, 1.0, size=n)), 1, 6).astype(int)

    lo = np.array([JARAK_KM[k][0] for k in kota], dtype=float)
    hi = np.array([JARAK_KM[k][1] for k in kota], dtype=float)
    jarak = rng.uniform(lo, hi)
    if scenario == "drift":
        jarak = jarak + rng.uniform(5, 15, size=n)  # pengembangan perumahan ke pinggiran
    jarak = jarak.round(1)

    umur = rng.integers(0, 41, size=n)

    harga_m2 = np.array([HARGA_M2[k] for k in kota], dtype=float)
    harga = (
        luas_tanah * harga_m2 * np.exp(-0.012 * jarak)
        + luas_bangunan * 4_500_000 * (1 - 0.012 * umur)
        + kamar_mandi * 25_000_000
    )
    if scenario == "drift":
        harga = harga * 1.18  # concept drift: harga pasar naik
    harga = harga * rng.lognormal(mean=0.0, sigma=0.12, size=n)  # noise pasar
    harga = (np.round(harga / 1_000_000) * 1_000_000).astype("int64")  # bulatkan ke juta

    return pd.DataFrame(
        {
            "luas_tanah_m2": luas_tanah.astype(int),
            "luas_bangunan_m2": luas_bangunan.astype(int),
            "kamar_tidur": kamar_tidur,
            "kamar_mandi": kamar_mandi,
            "jarak_ke_pusat_km": jarak,
            "umur_bangunan_tahun": umur,
            "kota": kota,
            "harga": harga,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--scenario", default="baseline", choices=["baseline", "live", "drift"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="data/train.csv")
    args = ap.parse_args()

    df = generate(args.n, args.scenario, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ {len(df)} baris ({args.scenario}) → {out}")
    print(df.describe(include="all").T[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()

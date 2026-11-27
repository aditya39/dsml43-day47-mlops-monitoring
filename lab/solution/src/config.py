"""src/config.py — baca config.yaml, lalu izinkan override dari environment variable.

Kenapa begini? Supaya angka (path, threshold, alamat server) tidak di-hardcode di kode,
dan laptop / container / CI bisa memakai nilai berbeda tanpa mengubah satu baris kode pun.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # folder lab (tempat config.yaml)

FEATURES = [
    "luas_tanah_m2",
    "luas_bangunan_m2",
    "kamar_tidur",
    "kamar_mandi",
    "jarak_ke_pusat_km",
    "umur_bangunan_tahun",
    "kota",
]
NUMERIC_FEATURES = FEATURES[:-1]
TARGET = "harga"
KOTA = [
    "Jakarta Selatan",
    "Jakarta Barat",
    "Jakarta Timur",
    "Depok",
    "Tangerang Selatan",
    "Bekasi",
    "Bogor",
]


@dataclass(frozen=True)
class Config:
    model_name: str
    experiment: str
    data_path: Path
    test_size: float
    min_rows: int
    model_params: dict
    random_state: int
    max_mae: float
    tracking_uri: str


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.getenv("CONFIG_PATH", ROOT / "config.yaml"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = dict(raw["model"])
    random_state = int(model.pop("random_state"))
    return Config(
        model_name=raw["project"]["model_name"],
        experiment=raw["project"]["experiment"],
        data_path=ROOT / raw["data"]["path"],
        test_size=float(raw["data"]["test_size"]),
        min_rows=int(raw["data"]["min_rows"]),
        model_params=model,
        random_state=random_state,
        # environment variable menang atas config.yaml
        max_mae=float(os.getenv("MAX_MAE", raw["quality_gate"]["max_mae"])),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", raw["mlflow"]["tracking_uri"]),
    )

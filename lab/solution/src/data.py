"""src/data.py — ambil data, VALIDASI, lalu split.

Checkpoint pertama pipeline: data rusak dihentikan di sini, bukan setelah model jadi.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURES, KOTA, TARGET


class DataValidationError(ValueError):
    """Data tidak layak dipakai untuk training."""


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ada. Buat dulu: python make_dataset.py")
    return pd.read_csv(path)


def validate(df: pd.DataFrame, min_rows: int = 500) -> pd.DataFrame:
    """Cek skema & isi data. Kalau ada masalah -> DataValidationError (pipeline berhenti)."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Kolom wajib tidak ada: {missing}")

    problems = []
    if len(df) < min_rows:
        problems.append(f"baris cuma {len(df)} (< {min_rows})")
    nulls = df[FEATURES + [TARGET]].isna().sum()
    if nulls.any():
        problems.append(f"ada nilai kosong: {nulls[nulls > 0].to_dict()}")
    if (df[["luas_tanah_m2", "luas_bangunan_m2"]] <= 0).any().any():
        problems.append("ada luas <= 0")
    if (df[TARGET] <= 0).any():
        problems.append("ada harga <= 0")
    unknown = sorted(set(df["kota"].dropna()) - set(KOTA))
    if unknown:
        problems.append(f"kota tidak dikenal: {unknown}")

    if problems:
        raise DataValidationError("; ".join(problems))
    return df


def split(df: pd.DataFrame, test_size: float, random_state: int):
    """Train/test split dengan seed tetap -> test set SAMA untuk setiap run (adil dibandingkan)."""
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=test_size, random_state=random_state
    )


def file_hash(path: str | Path) -> str:
    """Sidik jari file data (12 karakter pertama SHA-256) -> 'dilatih pakai data yang mana?'"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

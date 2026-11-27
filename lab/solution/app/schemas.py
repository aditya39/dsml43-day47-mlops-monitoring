"""app/schemas.py — kontrak input/output API (Pydantic).

Request /predict SAMA PERSIS dengan Day 46: client tidak perlu tahu bahwa sekarang
modelnya diambil dari MLflow Registry. Response menambah request_id & latency_ms.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Kota = Literal[
    "Jakarta Selatan",
    "Jakarta Barat",
    "Jakarta Timur",
    "Depok",
    "Tangerang Selatan",
    "Bekasi",
    "Bogor",
]


class HouseFeatures(BaseModel):
    """Satu rumah = satu baris fitur. Nama & tipe field = kolom training."""

    luas_tanah_m2: int = Field(gt=0, le=10_000, examples=[120])
    luas_bangunan_m2: int = Field(gt=0, le=10_000, examples=[100])
    kamar_tidur: int = Field(ge=1, le=20, examples=[3])
    kamar_mandi: int = Field(ge=1, le=20, examples=[2])
    jarak_ke_pusat_km: float = Field(ge=0, le=200, examples=[25.0])
    umur_bangunan_tahun: int = Field(ge=0, le=200, examples=[5])
    kota: Kota = Field(examples=["Depok"])


class PredictionResponse(BaseModel):
    harga_prediksi: int = Field(description="Prediksi harga (Rupiah)")
    model_version: str = Field(description="Versi model di MLflow Registry yang menjawab")
    request_id: str = Field(description="ID untuk mencari baris log request ini")
    latency_ms: float

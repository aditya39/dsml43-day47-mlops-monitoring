"""src/evaluate.py — ukur model + QUALITY GATE (gerbang mutu sebelum registry)."""
from __future__ import annotations

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


class QualityGateError(Exception):
    """Model tidak lolos gerbang mutu -> tidak boleh didaftarkan ke registry."""


def evaluate(model, X, y) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def quality_gate(metrics: dict[str, float], max_mae: float) -> None:
    """Gerbang mutu: MAE di atas batas = raise (BUKAN print), supaya pipeline benar-benar berhenti."""
    if metrics["mae"] > max_mae:
        mae, batas = format_rupiah(metrics["mae"]), format_rupiah(max_mae)
        raise QualityGateError(f"MAE {mae} > batas {batas}")


def format_rupiah(x: float) -> str:
    """251_123_456 -> 'Rp 251,1 jt'   |   1_710_000_000 -> 'Rp 1,71 M'"""
    if abs(x) >= 1e9:
        return f"Rp {x / 1e9:.2f} M".replace(".", ",")
    return f"Rp {x / 1e6:.1f} jt".replace(".", ",")

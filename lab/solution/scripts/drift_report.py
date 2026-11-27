"""scripts/drift_report.py — bandingkan distribusi data training vs data "produksi".

    python scripts/drift_report.py --current data/live_normal.csv
    python scripts/drift_report.py --current data/live_drift.csv
    python scripts/drift_report.py --current logs/api.jsonl      # fitur diambil dari log JSON API

Per fitur: KS test (numerik) / chi-square (kategori) + PSI.
Status: DRIFT kalau PSI >= 0.25 | WASPADA kalau PSI >= 0.10 atau p < 0.05 | selain itu AMAN.
Exit code 1 kalau ada fitur DRIFT -> bisa dipakai sebagai pemicu retraining / alert.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supaya bisa import src.*

from src.config import FEATURES, NUMERIC_FEATURES, TARGET, load_config  # noqa: E402
from src.evaluate import format_rupiah  # noqa: E402

PSI_DRIFT, PSI_WARN, P_VALUE = 0.25, 0.10, 0.05


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: bagi data training jadi 10 keranjang, bandingkan proporsinya."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-4, None)  # hindari log(0)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    ref_pct = np.clip(ref_pct, 1e-4, None)  # hindari log(0)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status_of(psi_value: float, p_value: float) -> str:
    if psi_value >= PSI_DRIFT:
        return "DRIFT"
    if psi_value >= PSI_WARN or p_value < P_VALUE:
        return "WASPADA"
    return "AMAN"


def read_current(path: Path) -> pd.DataFrame:
    """CSV biasa, atau log JSON API (satu baris per request; baris non-JSON dilewati)."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raw = path.read_bytes()
    # PowerShell 5 menyimpan hasil '>' sebagai UTF-16 -> tangani juga
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig", "replace")
    rows = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "prediction" and "features" in entry:
            rows.append(entry["features"])
    if not rows:
        raise SystemExit(f"Tidak ada baris log 'prediction' di {path}")
    return pd.DataFrame(rows)


def model_check(cur: pd.DataFrame) -> None:
    """Kalau data punya label (harga asli), bandingkan MAE champion: test set vs data ini."""
    import mlflow  # noqa: PLC0415 — hanya dibutuhkan di sini

    from src.registry import alias_version, get_client, version_mae  # noqa: PLC0415

    cfg = load_config()
    client = get_client(cfg)
    version = alias_version(client, cfg.model_name, "champion")
    if version is None:
        print("(cek MAE dilewati: belum ada @champion)")
        return
    model = mlflow.pyfunc.load_model(f"models:/{cfg.model_name}/{version}")
    mae_now = float(np.mean(np.abs(model.predict(cur[FEATURES]) - cur[TARGET])))
    mae_ref = version_mae(client, cfg.model_name, version)
    print(f"MAE champion v{version}: test set {format_rupiah(mae_ref)} -> data ini "
          f"{format_rupiah(mae_now)} ({mae_now / mae_ref:.2f}x)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="data/train.csv")
    ap.add_argument("--current", required=True, help="CSV data produksi atau file log .jsonl")
    ap.add_argument("--no-model", action="store_true", help="lewati cek MAE via MLflow")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    cur = read_current(Path(args.current))
    print(f"Referensi: {args.reference} ({len(ref)} baris)  vs  sekarang: {args.current} ({len(cur)} baris)\n")
    print(f"{'fitur':<22} {'rata-rata ref -> sekarang':<28} {'p-value':>9} {'PSI':>7}  status")
    print("-" * 78)

    n_drift = n_warn = 0
    for col in FEATURES:
        if col in NUMERIC_FEATURES:
            p_value = ks_2samp(ref[col], cur[col]).pvalue
            value = psi(ref[col], cur[col])
            change = (cur[col].mean() / ref[col].mean() - 1) * 100
            shift = f"{ref[col].mean():.1f} -> {cur[col].mean():.1f} ({change:+.0f}%)"
        else:
            table = pd.crosstab(np.r_[np.zeros(len(ref)), np.ones(len(cur))],
                                np.r_[ref[col].to_numpy(), cur[col].to_numpy()])
            p_value = chi2_contingency(table).pvalue
            value = psi_categorical(ref[col], cur[col])
            shift = "(kategori: proporsi kota)"
        status = status_of(value, p_value)
        n_drift += status == "DRIFT"
        n_warn += status == "WASPADA"
        print(f"{col:<22} {shift:<28} {p_value:>9.2g} {value:>7.3f}  {status}")

    print("-" * 78)
    print(f"Ringkasan: {n_drift} fitur DRIFT, {n_warn} WASPADA")
    if TARGET in cur.columns and not args.no_model:
        try:
            model_check(cur)
        except Exception as err:  # noqa: BLE001
            print(f"(cek MAE dilewati: {err.__class__.__name__}: {str(err)[:100]})")
    return 1 if n_drift else 0


if __name__ == "__main__":
    sys.exit(main())

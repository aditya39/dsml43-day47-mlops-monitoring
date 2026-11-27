"""src/registry.py — helper kecil untuk MLflow Model Registry (dipakai train/promote/rollback)."""
from __future__ import annotations

import mlflow
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import Config
from src.evaluate import format_rupiah


def get_client(cfg: Config) -> MlflowClient:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return MlflowClient()


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """Versi yang sedang ditunjuk alias (mis. @champion), atau None kalau alias belum ada."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def version_mae(client: MlflowClient, name: str, version: str) -> float | None:
    """MAE (test set) dari run yang menghasilkan versi model ini."""
    mv = client.get_model_version(name, version)
    return client.get_run(mv.run_id).data.metrics.get("mae")


def print_registry(client: MlflowClient, name: str) -> None:
    """Tabel ringkas: versi, MAE, alias. Alias dibaca dari registered model (sumber kebenaran)."""
    aliases = client.get_registered_model(name).aliases  # {"champion": "1", ...}
    by_version: dict[str, list[str]] = {}
    for alias, ver in aliases.items():
        by_version.setdefault(str(ver), []).append("@" + alias)
    versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
    print(f"\nRegistry '{name}':")
    for mv in versions:
        mae = client.get_run(mv.run_id).data.metrics.get("mae")
        mae_txt = format_rupiah(mae) if mae is not None else "-"
        print(f"  v{mv.version:<3} MAE {mae_txt:<14} {' '.join(sorted(by_version.get(str(mv.version), [])))}")


def reload_api(api_url: str, expected_version: str) -> bool:
    """Minta API memuat ulang @champion, lalu pastikan /health melaporkan versi yang diharapkan."""
    try:
        requests.post(f"{api_url}/admin/reload", timeout=60).raise_for_status()
        served = requests.get(f"{api_url}/health", timeout=10).json().get("model_version")
    except requests.RequestException as err:
        print(f"[PERINGATAN] API di {api_url} tidak bisa dihubungi ({err.__class__.__name__}).")
        print("             Alias sudah dipindah; API akan memuatnya saat start/reload berikutnya.")
        return True
    if str(served) != str(expected_version):
        print(f"[GAGAL] API melayani v{served}, padahal seharusnya v{expected_version}.")
        return False
    print(f"[OK] API sekarang melayani v{served}  (cek: {api_url}/health)")
    return True

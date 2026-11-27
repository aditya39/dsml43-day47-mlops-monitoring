"""app/model_loader.py — ambil model @champion dari MLflow Model Registry (bukan file .joblib)."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

# Matikan telemetri anonim MLflow & progress bar download (supaya log tetap JSON bersih).
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

MODEL_NAME = os.getenv("MODEL_NAME", "harga-rumah")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

log = logging.getLogger("api.model")


@dataclass
class LoadedModel:
    model: mlflow.pyfunc.PyFuncModel
    version: str
    run_id: str


def load_champion(name: str = MODEL_NAME,
                  alias: str = MODEL_ALIAS) -> LoadedModel:
    client = MlflowClient()  # alamat server dari env MLFLOW_TRACKING_URI
    # 1) "@champion itu versi berapa?"
    mv = client.get_model_version_by_alias(name, alias)
    # 2) muat versi PERSIS itu (bukan "yang terbaru")
    model = mlflow.pyfunc.load_model(f"models:/{name}/{mv.version}")
    return LoadedModel(model=model, version=str(mv.version), run_id=mv.run_id)


class ModelStore:
    """Menyimpan model yang sedang melayani; bisa diganti saat runtime (reload) tanpa restart."""

    def __init__(self) -> None:
        self.current: LoadedModel | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def reload(self) -> tuple[str | None, str]:
        new = load_champion()  # kalau gagal -> exception, model lama tetap melayani
        with self._lock:
            old = self.current
            self.current = new
            self.last_error = None
        return (old.version if old else None), new.version

    def try_load(self) -> None:
        """Dipanggil saat startup. Registry kosong / MLflow mati -> API tetap hidup (503 di /predict)."""
        try:
            _, version = self.reload()
            log.info("model_loaded", extra={"fields": {"model_name": MODEL_NAME, "alias": MODEL_ALIAS,
                                                       "model_version": version}})
        except Exception as err:  # noqa: BLE001 — sengaja: API harus tetap bisa start
            self.last_error = f"{err.__class__.__name__}: {err}"[:300]
            log.error("model_load_failed", extra={"fields": {"model_name": MODEL_NAME, "alias": MODEL_ALIAS,
                                                             "error": self.last_error}})

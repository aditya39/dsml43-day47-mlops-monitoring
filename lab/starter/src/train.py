"""src/train.py — pipeline: data -> validasi -> train -> evaluasi -> QUALITY GATE -> registry.

Jalankan dari folder lab (venv aktif, server MLflow hidup):
    python -m src.train                                              # pakai config.yaml
    python -m src.train --n-estimators 200 --max-features 0.5        # eksperimen lain
    python -m src.train --max-depth 3                                # contoh model yang DITOLAK gate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import load_config
from src.data import DataValidationError, file_hash, load_data, split, validate
from src.evaluate import QualityGateError, evaluate, format_rupiah, quality_gate
from src.registry import alias_version, get_client

warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")

# Model disimpan dengan format "skops" (aman, bukan pickle). Pohon RandomForest perlu
# dinyatakan "dipercaya" secara eksplisit -> kita sendiri yang membuat file-nya.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def build_pipeline(params: dict, random_state: int) -> Pipeline:
    """Preprocessing + model dalam SATU objek -> disimpan sebagai SATU artifact (anti training-serving skew)."""
    prep = ColumnTransformer(
        [("kota", OneHotEncoder(handle_unknown="ignore"), ["kota"])], remainder="passthrough"
    )
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    return Pipeline([("preprocess", prep), ("model", model)])


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _number(text: str):
    """'0.5' -> 0.5, '3' -> 3, 'sqrt' -> 'sqrt' (untuk argumen CLI)."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Latih model harga rumah + catat ke MLflow")
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--max-features", type=_number)
    ap.add_argument("--max-mae", type=float, help="override batas quality gate (Rupiah)")
    ap.add_argument("--run-name", help="nama run di MLflow (default: dibuat dari parameter)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    params = dict(cfg.model_params)
    for key in ("n_estimators", "max_depth", "max_features"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    max_mae = args.max_mae or cfg.max_mae
    run_name = args.run_name or (
        f"rf{params['n_estimators']}-depth{params['max_depth']}-mf{params['max_features']}"
    )

    # 1-2. ambil + validasi data (berhenti di sini kalau datanya rusak)
    try:
        df = validate(load_data(cfg.data_path), min_rows=cfg.min_rows)
    except (DataValidationError, FileNotFoundError) as err:
        print(f"[DATA GAGAL] {err}")
        return 1
    X_train, X_test, y_train, y_test = split(df, cfg.test_size, cfg.random_state)

    client = get_client(cfg)
    mlflow.set_experiment(cfg.experiment)
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # 3. catat SEMUA yang dibutuhkan untuk mengulang run ini
            # TODO 2a — catat parameter & tag ke MLflow:
            #   mlflow.log_params({**params, "random_state": cfg.random_state,
            #                      "test_size": cfg.test_size, "n_rows": len(df)})
            #   mlflow.set_tags({"git_sha": git_sha(), "data_hash": file_hash(cfg.data_path),
            #                    "max_mae": max_mae})

            # 4-5. latih + ukur di test set yang sama untuk setiap run
            pipe = build_pipeline(params, cfg.random_state).fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)
            # TODO 2b — catat metrik (mae, rmse, r2) ke MLflow: mlflow.log_metrics(metrics)
            print(f"\nRun '{run_name}': MAE {format_rupiah(metrics['mae'])} | "
                  f"RMSE {format_rupiah(metrics['rmse'])} | R2 {metrics['r2']:.3f}")

            # 6. QUALITY GATE: gagal -> exception -> run FAILED, registry tidak pernah melihatnya
            quality_gate(metrics, max_mae)
            mlflow.set_tag("quality_gate", "passed")

            # 7. lolos gate -> simpan model + daftarkan versi baru di registry
            # TODO 4 — ganti baris raise di bawah dengan:
            #   info = mlflow.sklearn.log_model(
            #       pipe,
            #       name="model",
            #       registered_model_name=cfg.model_name,
            #       input_example=X_train.head(3),
            #       signature=infer_signature(X_train, pipe.predict(X_train)),
            #       skops_trusted_types=TRUSTED_TYPES,
            #   )
            raise NotImplementedError("TODO 4: simpan & daftarkan model (src/train.py)")
    except QualityGateError as err:
        client.set_tag(run.info.run_id, "quality_gate", "failed")
        print(f"[GATE GAGAL] {err}")
        print("             Model TIDAK didaftarkan. Run tetap tercatat di MLflow (status FAILED).")
        return 1

    # 8. versi baru = @challenger. Kalau belum ada champion sama sekali, langsung jadi @champion.
    version = str(info.registered_model_version)
    client.set_registered_model_alias(cfg.model_name, "challenger", version)
    print(f"[GATE LOLOS] MAE <= {format_rupiah(max_mae)} -> terdaftar sebagai "
          f"{cfg.model_name} v{version} (@challenger)")
    if alias_version(client, cfg.model_name, "champion") is None:
        client.set_registered_model_alias(cfg.model_name, "champion", version)
        print(f"             Belum ada champion -> v{version} langsung jadi @champion")
    else:
        print("             Bandingkan & promosikan dengan: python scripts/promote.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

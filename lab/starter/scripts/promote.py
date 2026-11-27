"""scripts/promote.py — jadikan @challenger sebagai @champion (kalau terbukti lebih baik).

    python scripts/promote.py                     # @challenger -> @champion, cek MAE dulu
    python scripts/promote.py --version 3         # promosikan versi tertentu
    python scripts/promote.py --force --reason "hotfix"   # lewati cek MAE (tercatat!)

Champion lama disimpan sebagai @previous -> itulah tujuan rollback.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supaya bisa import src.*

from src.config import load_config  # noqa: E402
from src.evaluate import format_rupiah  # noqa: E402
from src.registry import alias_version, get_client, print_registry, reload_api, version_mae  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="versi yang dipromosikan (default: @challenger)")
    ap.add_argument("--force", action="store_true", help="promosikan walau MAE lebih buruk")
    ap.add_argument("--reason", default="MAE lebih baik di test set yang sama")
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    args = ap.parse_args()

    cfg = load_config()
    client = get_client(cfg)
    name = cfg.model_name
    champion = alias_version(client, name, "champion")
    candidate = args.version or alias_version(client, name, "challenger")
    if candidate is None:
        print("[GAGAL] Tidak ada @challenger. Latih model dulu: python -m src.train")
        return 1
    if candidate == champion:
        print(f"v{candidate} sudah @champion. Tidak ada yang berubah.")
        print_registry(client, name)
        return 0

    # Gerbang promosi: kandidat harus lebih baik dari champion di test set yang SAMA
    cand_mae = version_mae(client, name, candidate)
    champ_mae = version_mae(client, name, champion) if champion else None
    if champ_mae is not None:
        print(f"Champion  v{champion}: MAE {format_rupiah(champ_mae)}")
    print(f"Kandidat  v{candidate}: MAE {format_rupiah(cand_mae)}")
    if champ_mae is not None and cand_mae > champ_mae and not args.force:
        print("[DITOLAK] Kandidat lebih buruk dari champion. Pakai --force kalau yakin.")
        return 1

    client.set_registered_model_alias(name, "champion", candidate)
    if champion:
        client.set_registered_model_alias(name, "previous", champion)  # jalan pulang untuk rollback
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client.set_model_version_tag(name, candidate, "promoted_at", now)
    client.set_model_version_tag(name, candidate, "promotion_reason", args.reason)
    print(f"[OK] @champion: v{champion} -> v{candidate}")
    print_registry(client, name)
    return 0 if reload_api(args.api_url, candidate) else 1


if __name__ == "__main__":
    sys.exit(main())

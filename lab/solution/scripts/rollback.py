"""scripts/rollback.py — kembalikan @champion ke versi sebelumnya. Hitungan detik, tanpa build ulang.

    python scripts/rollback.py                    # @champion -> @previous
    python scripts/rollback.py --to 1 --reason "MAE live naik 3x"
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # supaya bisa import src.*

from src.config import load_config  # noqa: E402
from src.registry import alias_version, get_client, print_registry, reload_api  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", help="versi tujuan (default: @previous)")
    ap.add_argument("--reason", default="rollback manual")
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    args = ap.parse_args()

    cfg = load_config()
    client = get_client(cfg)
    name = cfg.model_name
    current = alias_version(client, name, "champion")
    target = args.to or alias_version(client, name, "previous")
    if target is None:
        print("[GAGAL] Tidak ada @previous. Sebutkan versinya: python scripts/rollback.py --to 1")
        return 1
    if target == current:
        print(f"v{target} sudah @champion. Tidak ada yang berubah.")
        return 0

    # Rollback = memindahkan label. Model lama masih utuh di registry, tidak ada yang di-build ulang.
    client.set_registered_model_alias(name, "champion", target)
    if current:
        client.set_registered_model_alias(name, "previous", current)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        client.set_model_version_tag(name, current, "rolled_back_at", now)
        client.set_model_version_tag(name, current, "rollback_reason", args.reason)
    print(f"[OK] ROLLBACK @champion: v{current} -> v{target}")
    print_registry(client, name)
    return 0 if reload_api(args.api_url, target) else 1


if __name__ == "__main__":
    sys.exit(main())

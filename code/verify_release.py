"""Verify versioned public assets, safely extract, and run offline replays.

Usage: python -B code/verify_release.py --assets ../release-assets
Optional --native --token-assets PATH performs native/token replay too.
No downloads or API requests are made. Install the stated dependencies first.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--native", action="store_true")
    p.add_argument("--token-assets", type=Path)
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "package-inventory.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    reports = []
    with tempfile.TemporaryDirectory(prefix="memory-audit-public-") as tmp:
        extracted = Path(tmp)
        for key, row in inventory["packages"].items():
            source = args.assets / row["file"]
            assert sha(source) == row["sha256"], (key, "archive identity")
            dest = extracted / key
            with zipfile.ZipFile(source) as z:
                assert z.testzip() is None
                for item in z.infolist():
                    n = Path(item.filename)
                    assert not n.is_absolute() and ".." not in n.parts and ":" not in item.filename
                    assert (item.external_attr >> 16) & 0o170000 != 0o120000, "Symlink not allowed"
                z.extractall(dest)
        # S8 links original selection records to the independently replayed S3
        # files. Public archive hashes differ from the original review archives.
        links = json.loads((extracted / "S8/evidence/S3-selection-link-map.json").read_text(encoding="utf-8"))
        mappings = links["links"]
        for link in mappings:
            assert sha(extracted / "S3" / link["S3_zip_member"]) == link["source_sha256"]
        assert len(mappings) == 450
        for key in inventory["packages"]:
            command = [args.python, "-B", "-X", "utf8", "verify_public.py"]
            if args.native and key == "S6":
                assert args.token_assets is not None
                command += ["--native-replay", "--token-assets", str(args.token_assets.resolve())]
            if args.native and key == "S7":
                command += ["--native-replay", "--agentdojo-root", str(extracted / "S6")]
            result = subprocess.run(command, cwd=extracted / key, env=env, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                raise RuntimeError(f"{key} verification failed")
            last = result.stdout.strip().splitlines()[-1]
            reports.append({"package": key, "exit_code": 0, "result": json.loads(last)})
            print(json.dumps({"package": key, "status": "PASS"}), flush=True)
    print(json.dumps({"status": "PASS", "packages": reports, "S3_S8_original_record_links": 450,
                      "native_replay_requested": args.native, "new_model_calls": 0}))


if __name__ == "__main__":
    main()

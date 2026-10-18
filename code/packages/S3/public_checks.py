"""Checks shared by the derived public packages. No model calls are made."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integrity():
    manifest = read(ROOT / "PUBLIC_MANIFEST.json")
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and not any(part.startswith(".runtime") for part in p.relative_to(ROOT).parts)}
    assert actual == set(manifest["files"]) | {"PUBLIC_MANIFEST.json"}, "Unexpected or missing public files"
    for name, item in manifest["files"].items():
        assert sha(ROOT / name) == item["sha256"], name
    return manifest


def original_subset(mapping):
    """Check retained original dependencies, explicitly report omitted old gates.

    This is NOT a pass for the original full-checkout freeze gate. A file that
    was intentionally transformed is checked against its recorded original hash
    AND its current public hash. Missing original dependencies are not fabricated.
    """
    manifest = read(ROOT / "PUBLIC_MANIFEST.json")
    checked = transformed = omitted = 0
    for name, expected in mapping.items():
        row = manifest["files"].get(name)
        if row is None:
            omitted += 1
            continue
        assert row.get("original_sha256", row["sha256"]) == expected, (name, "original identity")
        assert sha(ROOT / name) == row["sha256"], (name, "public identity")
        checked += 1
        transformed += row["sha256"] != expected
    return {"retained_dependencies_checked": checked, "transformed_sources": transformed,
            "excluded_original_dependencies": omitted, "original_full_freeze_gate_pass_claimed": False}


def activate_guard(extra_read_roots=()):
    """Disallow network/process access and credential reads; evidence is read-only.

    Optional dependency initialization may use .runtime-cache. This Python audit
    hook is an execution safeguard, not a native-extension security proof.
    """
    sys.dont_write_bytecode = True
    sys.dontwritebytecode = True
    for key in tuple(os.environ):
        if key.endswith(("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN")):
            os.environ.pop(key)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    scratch = ROOT / ".runtime-cache"
    scratch.mkdir(exist_ok=True)
    import tempfile
    tempfile.tempdir = str(scratch)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "torchinductor")
    allowed_reads = [ROOT, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    allowed_reads += [Path(x).resolve() for x in extra_read_roots]

    def guard(event, args):
        if event.startswith(("socket.", "subprocess.", "os.spawn")) or event in {"os.system", "os.startfile"}:
            raise PermissionError("Public offline verification blocks " + event)
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.name == ".env" or p.name.startswith(".env."):
                raise PermissionError("Credential file read blocked")
            if not any(p == root or root in p.parents for root in allowed_reads):
                raise PermissionError("Read outside extracted package and explicit runtime/assets blocked: " + str(p))
            mode, flags = args[1], args[2]
            writes = isinstance(mode, str) and any(c in mode for c in "wax+")
            writes |= isinstance(flags, int) and bool(flags & 0x703)
            if writes and p != scratch and scratch not in p.parents:
                raise PermissionError("Evidence write blocked: " + str(p))
        if event in {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod", "os.utime", "os.link", "os.symlink", "os.truncate"}:
            for value in args[:2 if event in {"os.rename", "os.link", "os.symlink"} else 1]:
                if not isinstance(value, (str, bytes, os.PathLike)):
                    raise PermissionError("Unknown mutation blocked")
                p = Path(os.fsdecode(value)).resolve()
                if p != scratch and scratch not in p.parents:
                    raise PermissionError("Evidence mutation blocked: " + event)
    sys.addaudithook(guard)
    for event in ("socket.connect", "subprocess.Popen"):
        try:
            sys.audit(event, None)
        except PermissionError:
            pass
        else:
            raise AssertionError("Guard self-test failed: " + event)


if __name__ == "__main__":
    result = integrity()
    print(json.dumps({"status": "PASS", "public_files_checked": len(result["files"])}))

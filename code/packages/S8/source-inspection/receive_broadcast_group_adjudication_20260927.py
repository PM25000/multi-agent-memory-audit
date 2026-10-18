"""本轮 C 收件修订：理由选填；原冻结源、答卷和统计方法保持不变。

仅适用已冻结的 20260923 包和已收到的 A/B。无网络或模型调用。
--check 只读重算并逐字节比较已归档的收件/分析产物。
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import prepare_broadcast_group_review as frozen

AMENDMENT = "broadcast-group-reasons-optional-20260927"
SOURCE_HASHES = {
    "A": "688af03928fd3e5bb613df4a4cfe18cfa73fd18feaa794cfa6f564130448cbab",
    "B": "4f91f4213a8c687eaf2f4ca9ff7e04c8886470cb915c0076204adca42b8ad999",
}
PACKAGE_HASH = "76e5a2c7b39f85b3092be4850d3b22dca10e0093adecdbe27faf4f4b9fa1ca23"
ORIGINAL_VALIDATE = frozen.validate


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw.lstrip("\ufeff"), object_pairs_hook=unique_object)


def validate_optional_reason(answer, out, allow_fixture=False):
    """保留原校验全部条件；C 的理由字符串允许为空。A/B 沿用原校验。"""
    assert not allow_fixture, "formal intake only"
    assert isinstance(answer, dict)
    if answer.get("reviewer") in {"A", "B"}:
        return ORIGINAL_VALIDATE(answer, out)
    assert answer.get("reviewer") == "C"
    assert not answer.get("demo") and not answer.get("test_fixture")
    assert answer.get("version") == frozen.VERSION
    p = frozen.payload_for(out, "C")
    assert answer.get("package_hash") == p["package_hash"] == PACKAGE_HASH
    assert answer.get("page_token") == p["page_token"]
    assert isinstance(answer.get("exported_at"), str) and answer["exported_at"]
    answers = answer.get("answers")
    assert isinstance(answers, dict)
    groups = {g["id"]: g for g in p["groups"]}
    assert set(answers) <= set(groups)
    for gid, a in answers.items():
        assert isinstance(a, dict)
        assert a.get("fact_status", "") in frozen.FACTS | {""}
        assert isinstance(a.get("note", ""), str)
        assert isinstance(a.get("needs", {}), dict)
        tasks = {t["id"] for t in p["cases"][groups[gid]["case_id"]]["tasks"]}
        assert set(a.get("needs", {})) <= tasks
        for value in a.get("needs", {}).values():
            assert isinstance(value, dict)
            assert value.get("status", "") in frozen.NEEDS | {""}
            assert isinstance(value.get("note", ""), str)
    assert isinstance(answer.get("declared_adjudicated"), bool)
    assert answer["declared_adjudicated"], "human completion declaration required"
    assert isinstance(answer.get("source_raw"), dict)
    assert set(answer["source_raw"]) == {"A", "B"}
    for role, raw in answer["source_raw"].items():
        assert isinstance(raw, str)
        data = raw.encode("utf-8")
        assert frozen.digest(data) == SOURCE_HASHES[role], "different A/B original"
        assert (out / "receipts" / f"{role}_{SOURCE_HASHES[role]}.json").read_bytes() == data
        child = loads(raw)
        assert child["reviewer"] == role
        ORIGINAL_VALIDATE(child, out)
    return p


def calculate(answer, out):
    # Scoped adapter: only the validator is substituted, never data or arithmetic.
    # On exit the frozen module is restored, including when validation fails.
    with patch.object(frozen, "validate", validate_optional_reason):
        return frozen.analyze_data(answer, out)


def self_test(answer, out):
    """验证修订仍拒收伪答卷，且缺失裁决不会被自动填补。"""
    for key, bad in [("page_token", "wrong"), ("package_hash", "wrong"),
                     ("demo", True), ("test_fixture", True),
                     ("declared_adjudicated", False)]:
        x = deepcopy(answer)
        x[key] = bad
        try:
            validate_optional_reason(x, out)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted invalid {key}")
    x = deepcopy(answer)
    x["source_raw"]["A"] += " "
    try:
        validate_optional_reason(x, out)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted changed A")
    try:
        loads('{"x": 1, "x": 2}')
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted duplicate keys")
    # A deleted disputed decision must remain unresolved, not use old reference.
    ab = {r: loads(answer["source_raw"][r]) for r in ("A", "B")}
    for gid, group in answer["answers"].items():
        for tid, value in group.get("needs", {}).items():
            if value.get("status") and ab["A"]["answers"][gid]["needs"][tid]["status"] != ab["B"]["answers"][gid]["needs"][tid]["status"]:
                x = deepcopy(answer)
                x["answers"][gid]["needs"][tid]["status"] = ""
                with patch.object(frozen, "validate", validate_optional_reason):
                    rows, _ = frozen.resolved(x, out)
                assert rows[gid]["needs"][tid] == ""
                assert rows[gid]["provenance"][tid] == "unresolved"
                return
    raise AssertionError("no disputed decision available to test")


def put(path, raw, check):
    if check or path.exists():
        assert path.read_bytes() == raw, f"artifact drift: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answer", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = frozen.OUT
    frozen.check(out)
    raw = args.answer.read_bytes()
    answer = loads(raw.decode("utf-8-sig"))
    validate_optional_reason(answer, out)
    self_test(answer, out)
    result = calculate(answer, out)
    rows = result["resolved_with_provenance"]
    counts = Counter(v for r in rows.values() for v in r["needs"].values())
    provenance = Counter(v for r in rows.values() for k, v in r["provenance"].items() if k != "fact_status")
    original = {r: loads(answer["source_raw"][r]) for r in ("A", "B")}
    overrides, pending, blanks = [], [], []
    for gid, row in rows.items():
        for tid, status in row["needs"].items():
            va, vb = (original[r]["answers"][gid]["needs"][tid]["status"] for r in ("A", "B"))
            explicit = answer["answers"].get(gid, {}).get("needs", {}).get(tid)
            if va == vb and explicit and explicit.get("status"):
                overrides.append({"group": gid, "task": tid, "A_B": va, "C": status})
            if status in {"", "uncertain"}:
                pending.append({"group": gid, "task": tid})
            if explicit is not None and not explicit.get("status"):
                blanks.append({"group": gid, "task": tid, "inherited": status, "provenance": row["provenance"][tid]})
    sha = frozen.digest(raw)
    receipt = {
        "amendment": AMENDMENT,
        "authorization": "2026-09-27 protocol amendment: written adjudication reasons are optional; missing reasons are not fabricated",
        "reason_requirement": "optional; no reasons invented or inserted",
        "package_hash": PACKAGE_HASH, "C_sha256": sha, "C_bytes": len(raw),
        "source_sha256": SOURCE_HASHES,
        "declared_adjudicated": answer["declared_adjudicated"],
        "groups": len(rows), "final_fact_counts": dict(Counter(r["fact_status"] for r in rows.values())),
        "final_need_counts": dict(counts), "need_provenance_counts": dict(provenance),
        "raw_independent_comparison": result["raw_independent_comparison"],
        "explicit_agreement_overrides": overrides, "pending": pending,
        "explicit_blank_entries": blanks,
        "notes_present": sum(bool(g.get("note", "").strip()) for g in answer["answers"].values()),
        "frozen_source_unchanged": True, "reference_updated": False,
        "new_model_calls": 0,
        "adapter_sha256": frozen.digest(Path(__file__).read_bytes()),
    }
    result["intake_amendment"] = AMENDMENT
    result["answer_sha256"] = sha
    dest = out / "analysis_reasons_optional_20260927" / sha
    put(out / "receipts" / f"C_{sha}.json", raw, args.check)
    put(dest / "receipt.json", frozen.pretty(receipt), args.check)
    put(dest / "summary.json", frozen.pretty(result), args.check)
    frozen.check(out)
    print(json.dumps({"check": "PASS" if args.check else "RECEIVED", "receipt": receipt,
                      "results": result["results"], "directory": str(dest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

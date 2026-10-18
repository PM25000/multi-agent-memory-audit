"""冻结新增广播事实组核验；只读检查、原件收件和裁决后离线设计分析。

No network/model calls. Never edits an existing necessity reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/results/review_revision_20260910"
OUT = BASE / "broadcast_group_review_20260923"
SOURCE = BASE / "unified_reference_20260922"
DATA = BASE / "fair_comparison_ids/datasets/test.json"
TEMPLATE = ROOT / "experiments/broadcast_group_review.html"
VERSION = "broadcast-group-review-1"
SEED = 20260923
MODELS = ("qwen", "glm", "gpt55_responses")
ALLOC = {2: 34, 3: 42, 4: 20}
SIZES = {2: 577, 3: 724, 4: 343}
FACTS = {"supported", "unsupported", "uncertain"}
NEEDS = {"required", "possible_only", "not_required", "uncertain"}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pretty(x):
    return (json.dumps(x, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(b):
    return hashlib.sha256(b).hexdigest()


def opaque(kind, *parts):
    return kind + digest(canonical([VERSION, *parts]))[:12].upper()


def make_frame():
    cases = {x["key"]: x for x in read(DATA) if x["split"] == "multi"}
    refs = read(SOURCE / "references.json")
    metrics = {m: {x["key"]: x["metrics"]["joint"] for x in read(SOURCE / "required_per_input" / f"{m}.json")} for m in MODELS}
    frame = []
    relations = set()
    for key, case in sorted(cases.items()):
        active = sorted(t["task_index"] for t in refs[key]["tasks"] if t["active"])
        memory = {x["id"] for x in case["entities"] if x["source"] == "memory"}
        selected = {m: {v["task_index"]: set(v["visible"]) & memory for v in metrics[m][key]["visibility"]} for m in MODELS}
        assert all(set(selected[m]) == set(active) for m in MODELS)
        for entity in sorted(memory):
            selectors = {m: [t for t in active if entity in selected[m][t]] for m in MODELS}
            added = {m: [t for t in active if t not in selectors[m]] if selectors[m] else [] for m in MODELS}
            if not any(added.values()):
                continue
            for recipients in added.values():
                relations.update((key, t, entity) for t in recipients)
            frame.append({"group_id": opaque("G", key, entity), "case_key": key, "entity_id": entity,
                          "active_tasks": active, "task_count": len(active), "selectors": selectors, "added": added})
    assert len(frame) == 1644 and len(relations) == 2976
    assert dict(Counter(x["task_count"] for x in frame)) == SIZES
    return frame, cases


def draw(frame):
    sample = []
    for h, n in ALLOC.items():
        pool = [x for x in frame if x["task_count"] == h]
        rng = random.Random(SEED + h)
        chosen = rng.sample(pool, n)
        rng.shuffle(chosen)
        for i, row in enumerate(chosen):
            sample.append({**row, "batch": 1 if i < n // 2 else 2,
                           "inclusion_probability": n / len(pool), "design_weight": len(pool) / n})
    assert len(sample) == 96 and sum(x["task_count"] for x in sample) == 274
    return sorted(sample, key=lambda x: (x["batch"], x["case_key"], x["entity_id"]))


def public_content(sample, cases):
    public_cases, groups = {}, []
    for row in sample:
        case = cases[row["case_key"]]
        cid = opaque("C", row["case_key"])
        codes = {x["id"]: f"F{i:02}" for i, x in enumerate(case["entities"], 1)}
        public_cases[cid] = {"id": cid, "memory": case["item"]["memory_context"], "request": case["item"]["user_input"],
            "tasks": [{"id": f"T{t['task_index'] + 1}", "name": t["name"], "quotes": t["request_quotes"]}
                      for t in case["fixed_tasks"] if t["task_index"] in row["active_tasks"]],
            "entities": [{"code": codes[x["id"]], "label": x["label"], "value": x["value"], "source": x["source"], "quote": x["quote"]} for x in case["entities"]]}
        fact = next(x for x in case["entities"] if x["id"] == row["entity_id"])
        groups.append({"id": row["group_id"], "case_id": cid, "batch": row["batch"], "fact_code": codes[fact["id"]],
                       "label": fact["label"], "value": fact["value"], "quote": fact["quote"]})
    return {"version": VERSION, "groups": groups, "cases": public_cases,
            "group_count": len(groups), "need_count": 274, "batches": 2, "budget_minutes_total": 120}


def page_payload(core, role, package_hash):
    p = json.loads(canonical(core))
    if role in ("A", "B"):
        rng = random.Random(SEED + (101 if role == "A" else 202))
        ordered = []
        for batch in (1, 2):
            by_case = defaultdict(list)
            for g in p["groups"]:
                if g["batch"] == batch:
                    by_case[g["case_id"]].append(g)
            ids = sorted(by_case)
            rng.shuffle(ids)
            for cid in ids:
                rng.shuffle(by_case[cid])
                ordered.extend(by_case[cid])
        p["groups"] = ordered
    p.update(reviewer=role, package_hash=package_hash, demo=False)
    if role == "C":
        p["source_tokens"] = {r: page_payload(core, r, package_hash)["page_token"] for r in ("A", "B")}
    p["page_token"] = digest(canonical(p))
    return p


def render(payload):
    packed = canonical(payload).decode("utf-8").replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("__PACKAGE_JSON__", packed).encode("utf-8")


def zip_bytes(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    return out.getvalue()


def make_files():
    frame, cases = make_frame()
    sample = draw(frame)
    core = public_content(sample, cases)
    package_hash = digest(canonical(core))
    sources = [DATA, SOURCE / "references.json", TEMPLATE, Path(__file__).resolve(), ROOT / "experiments/check_broadcast_group_review.py"] + [SOURCE / "required_per_input" / f"{m}.json" for m in MODELS]
    protocol = {
        "version": VERSION, "frozen_date": "2026-09-23", "seed": SEED, "package_hash": package_hash,
        "frame": "Input-memory-record groups with any extra recipient under at least one frozen joint selector; all active task relations reviewed",
        "frame_groups": 1644, "extra_relations_union": 2976, "sample_groups": 96, "need_judgments": 274,
        "strata": {str(h): {"population": SIZES[h], "sample": ALLOC[h], "pi": ALLOC[h]/SIZES[h]} for h in ALLOC},
        "design": "Independent SRS without replacement within active-task-count strata; seed+task_count; equal halves per batch; one final analysis",
        "budget": "User selected approximately two hours per reviewer, divisible into sessions; no forced answers, no replacement or adaptive sample extension",
        "primary_estimands": ["Additional non-required recipient-record transmissions / 150 inputs", "Correct-selected spillover share among additional non-required transmissions"],
        "endpoints": {"lower": "required and possible_only treated needed", "upper": "only required treated needed"},
        "estimators": "HT total sum_h N_h mean_h(y); ratio of weighted correct-spillover / weighted unnecessary totals; use entire group vectors",
        "uncertainty": "Approximate design normal intervals with FPC and ratio linearization only when all sampled values identified; also simultaneous bounded Hoeffding intervals for 12 totals (3 models x 2 endpoints x 2 totals), alpha=.05/12",
        "conservative_radius": "sqrt(log(2/(.05/12))/2 * sum_h N_h^2*(h-1)^2/n_h); no finite population improvement claimed",
        "unknown_policy": "Missing/uncertain needs expand to {0,1}; unsupported/uncertain/missing fact grounding expands whole group vector; enumerate up to 16 vectors. Bound each weighted total then add sampling radius. Never drop missing groups or impute old labels.",
        "fact_policy": "Required fact-grounding question per group. Unsupported grounding does not redefine the record as zero exposure; remains unknown for semantic primary results.",
        "agreement_policy": "Carry through matching determinate A/B values with provenance; uncertain/missing/disagreement requires explicit author decision. C sees A/B and is not independent. Overrides need reasons.",
        "stopping": "Fixed 96 groups, two batches; no interim effect feedback; budget exhausted -> export missing; no optional stopping for significance. Claims follow achieved precision.",
        "limits": "No full-reference accuracy, true field completeness, whole-needed-record recall, independent residual composition or natural-world prevalence. Old96 stays separate; no pooled unweighted proportions.",
        "blinding": ["model", "selections", "D/U", "prior labels", "old answers", "weights", "paper outcomes"],
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in sources},
        "frame_sha256": digest(canonical(frame)), "sample_sha256": digest(canonical(sample)),
        "statistical_reference": "https://arxiv.org/pdf/1309.4029 (Lemma 1.1 and Proposition 1.2); independent-stratum MGF combination and Bonferroni",
    }
    radius = math.sqrt(math.log(480)/2 * sum(SIZES[h]**2*(h-1)**2/ALLOC[h] for h in ALLOC))
    protocol["planning_simultaneous_radius_per_input"] = radius / 150
    files = {"private/frame.json": pretty(frame), "private/selection.json": pretty(sample),
             "private/core.json": pretty(core), "protocol.json": pretty(protocol)}
    for role in ("A", "B", "C"):
        p = page_payload(core, role, package_hash)
        files[f"{'author' if role == 'C' else 'public'}/{'adjudicator' if role == 'C' else 'reviewer_' + role}.html"] = render(p)
        if role in ("A", "B"):
            instructions = f"""任务信息必要性核验 · {role}
只把本包交给标注者 {role}；请勿在独立作答前交换答案。
解压后双击 核验工作台_{role}.html，建议使用 Chrome 或 Edge。
共96组，每组核对一条信息的原文依据，并判断它对全部2–4个任务是否必要。
分两批，各48组、137个任务判断；总计274个任务判断和96个原文依据判断。
计划每人约两小时，可分次；实际速度因人而异。到时间仍未答的保留空白，不猜答案。
左侧完整原文是依据，右侧逐任务作答。可能需要表示任务条件本身不明确；无法判断表示你暂时不能判断。
页面自动保存在当前浏览器；换浏览器/电脑前导出备份。收到新版不要覆盖旧答卷。
完成或时间到后勾选真实的独立声明、填写是否参与旧核验，导出答卷JSON交回作者；HTML不是答卷。
未答可以导出；不要把演练答卷当正式答卷。不上传联网，无自动提交。
包标识：{package_hash}
""".encode("utf-8")
            members = {f"核验工作台_{role}.html": render(p), "填写说明.txt": instructions}
            members["文件校验.json"] = pretty({"version": VERSION, "reviewer": role, "package_hash": package_hash,
                                               "sha256": {n: digest(b) for n, b in members.items()}})
            files[f"sendable/必要性核验_{role}_20260923.zip"] = zip_bytes(members)
    demo_case = {"id": "DEMO_CASE", "memory": "虚构演练：小林的会议主题是项目进度，喜欢蓝色。", "request": "请起草项目进度会议议程，再推荐一款蓝色笔记本。",
                 "tasks": [{"id": "T1", "name": "起草会议议程", "quotes": ["请起草项目进度会议议程"]}, {"id": "T2", "name": "推荐笔记本", "quotes": ["再推荐一款蓝色笔记本"]}],
                 "entities": [{"code": "F01", "label": "颜色偏好", "value": "蓝色", "source": "memory", "quote": "喜欢蓝色"}]}
    demo = {"version": VERSION + "-demo", "reviewer": "A", "demo": True, "package_hash": "DEMO-NOT-RESEARCH", "page_token": "DEMO",
            "groups": [{"id": "DEMO_GROUP", "case_id": "DEMO_CASE", "batch": 1, "fact_code": "F01", "label": "颜色偏好", "value": "蓝色", "quote": "喜欢蓝色"}],
            "cases": {"DEMO_CASE": demo_case}, "group_count": 1, "need_count": 2, "batches": 1, "budget_minutes_total": 0}
    files["demo/practice.html"] = render(demo)
    author_demo = {**demo, "reviewer": "C", "source_tokens": {"A": "DEMO", "B": "DEMO"}}
    files["demo/adjudicator_practice.html"] = render(author_demo)
    for role, need in (("A", "not_required"), ("B", "uncertain")):
        files[f"demo/DEMO-{role}.json"] = pretty({"version": demo["version"], "reviewer": role, "demo": True,
            "package_hash": demo["package_hash"], "page_token": "DEMO", "exported_at": "2026-09-23T00:00:00Z",
            "declared_independent": True, "prior_participation": "no", "reviewer_code": "DEMO-FICTION-NOT-A-PERSON",
            "answers": {"DEMO_GROUP": {"fact_status": "supported", "needs": {"T1": {"status": need}, "T2": {"status": "required"}}, "note": "虚构测试答案，不能作为研究证据"}}})
    files["README.md"] = f"""# 任务信息核验工作台（作者入口）

已冻结96个信息组：34个双任务、42个三任务、20个四任务组；每人274个任务必要性判断及96个原文依据判断。两批各48组/137个任务判断。约两小时是工作预算，不保证完成时间或统计精度。未答保留缺失。

- [发给 A 的独立包](sendable/必要性核验_A_20260923.zip)
- [发给 B 的独立包](sendable/必要性核验_B_20260923.zip)
- [作者收件与裁决工作台](author/adjudicator.html)：导入两份原始JSON；显示独立判断，保留确定的一致项；分歧、未答和无法判断待真人处理。导出裁决答卷时内嵌原始文本以便保留原件。
- [演练页面](demo/practice.html)：纯虚构练习，不属于正式样本。

只向标注者发送其专属ZIP，勿发送作者页面或private目录。旧96项答卷不变；这次是新的定向概率审计，不能合并为未加权准确率。字段恢复仍为冻结方案诊断，本包不重新定义字段。

收件：`python -X utf8 experiments/prepare_broadcast_group_review.py --receive 答卷.json`

裁决后设计分析：`python -X utf8 experiments/prepare_broadcast_group_review.py --analyze 裁决答卷.json`

只读核验：`python -X utf8 experiments/prepare_broadcast_group_review.py --check`

收件原件按内容哈希保留；多个版本需明确选择，不按文件时间自动取最新。分析仅写本目录独立analysis子目录，不修改20260922参考或论文表。来源绑定变更另行人工处理。

包标识：`{package_hash}`。

统计使用按任务数分层的组级无放回抽样；权重、缺失处理、停止规则和限制见protocol.json。保守抽样区间方法依据 [Bardenet与Maillard，Lemma 1.1 / Proposition 1.2](https://arxiv.org/pdf/1309.4029)，对独立分层作有界总量组合，并对12个目标总量控制同时覆盖；另报明确标注为近似的设计区间。未收答案时不会生成研究结果。
""".encode("utf-8")
    files["index.html"] = ("<!doctype html><meta charset='utf-8'><title>核验工作台 · 作者入口</title><style>body{font:17px/1.8 'Microsoft YaHei',sans-serif;max-width:900px;margin:50px auto;padding:24px;background:#f5f7fa;color:#20352c}a{color:#22634e}li{margin:14px 0}small{color:#5b6870}</style><h1>核验工作台 · 作者入口</h1><p>先分发 A / B 独立包，收齐答卷后再裁决。每人两批，各48组，可分次保存。</p><ul><li><a href='sendable/必要性核验_A_20260923.zip'>下载 A 独立发送包</a></li><li><a href='sendable/必要性核验_B_20260923.zip'>下载 B 独立发送包</a></li><li><a href='author/adjudicator.html'>打开作者收件与裁决工作台</a></li><li><a href='demo/practice.html'>打开虚构演练页面</a></li></ul><p>请勿把作者页面或整个目录发给复核者。收到的是 JSON 答卷；HTML 是填写页面。</p><small>当前尚无本轮人工答卷或新模型结果。旧96项材料保持不变。</small>").encode("utf-8")
    files["manifest.json"] = pretty({"version": VERSION, "package_hash": package_hash, "files": {n: digest(b) for n, b in sorted(files.items())}})
    return files, protocol


def build(out, preview=False):
    files, protocol = make_files()
    if out.exists() and not preview:
        check(out)
        return
    for name, content in files.items():
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    print(json.dumps({"status": "preview" if preview else "frozen", "path": str(out), "package_hash": protocol["package_hash"], "groups": 96, "needs": 274}, ensure_ascii=False))


def check(out):
    manifest = read(out / "manifest.json")
    for name, h in manifest["files"].items():
        assert digest((out / name).read_bytes()) == h, name
    files, _ = make_files()
    assert (out / "manifest.json").read_bytes() == files["manifest.json"], "source/package drift; freeze a new version, never overwrite"
    for name, raw in files.items():
        assert (out/name).read_bytes() == raw, name
    for role in ("A", "B"):
        with zipfile.ZipFile(out / f"sendable/必要性核验_{role}_20260923.zip") as z:
            assert len(z.namelist()) == 3
            payload = extract_payload(z.read(f"核验工作台_{role}.html").decode("utf-8"))
            assert payload["reviewer"] == role
            def keys(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        yield k
                        yield from keys(v)
                elif isinstance(x, list):
                    for v in x:
                        yield from keys(v)
            assert not set(keys(payload)) & {"selectors", "added", "design_weight", "needed", "reference", "bindings", "a", "b", "case_key", "entity_id", "stratum"}
    print(json.dumps({"check": "PASS", "package_hash": manifest["package_hash"], "frozen_files": len(files)}, ensure_ascii=False))


def extract_payload(html):
    return json.loads(html.split('<script id="package" type="application/json">', 1)[1].split("</script>", 1)[0])


def payload_for(out, role):
    p = out / ("author/adjudicator.html" if role == "C" else f"public/reviewer_{role}.html")
    return extract_payload(p.read_text(encoding="utf-8"))


def validate(answer, out, allow_fixture=False):
    assert isinstance(answer, dict)
    assert not answer.get("demo") and (allow_fixture or not answer.get("test_fixture")), "practice/test answer cannot be received as research"
    assert answer.get("version") == VERSION
    role = answer.get("reviewer")
    assert role in {"A", "B", "C"}
    p = payload_for(out, role)
    assert answer.get("package_hash") == p["package_hash"]
    assert answer.get("page_token") == p["page_token"]
    assert isinstance(answer.get("exported_at"), str) and answer["exported_at"]
    answers = answer.get("answers")
    assert isinstance(answers, dict)
    groups = {g["id"]: g for g in p["groups"]}
    assert set(answers) <= set(groups), "unknown group id"
    for gid, a in answers.items():
        assert isinstance(a, dict)
        assert a.get("fact_status", "") in FACTS | {""}
        assert isinstance(a.get("note", ""), str) and isinstance(a.get("needs", {}), dict)
        task_ids = {t["id"] for t in p["cases"][groups[gid]["case_id"]]["tasks"]}
        assert set(a.get("needs", {})) <= task_ids
        for x in a.get("needs", {}).values():
            assert isinstance(x, dict) and x.get("status", "") in NEEDS | {""}
            assert isinstance(x.get("note", ""), str)
        if role == "C" and (a.get("fact_status") or any(x.get("status") for x in a.get("needs", {}).values())):
            assert a.get("note", "").strip(), "adjudication or override requires a reason"
    if role in {"A", "B"}:
        assert isinstance(answer.get("declared_independent"), bool)
        assert answer.get("prior_participation", "") in {"", "yes", "no", "unsure"}
        assert isinstance(answer.get("reviewer_code", ""), str)
    else:
        assert isinstance(answer.get("declared_adjudicated"), bool)
        assert isinstance(answer.get("source_raw", {}), dict)
        for who, raw in answer.get("source_raw", {}).items():
            assert who in {"A", "B"} and isinstance(raw, str)
            child = json.loads(raw.lstrip("\ufeff"))
            assert child["reviewer"] == who
            validate(child, out, allow_fixture)
    return p


def resolved(answer, out, allow_fixture=False):
    p = validate(answer, out, allow_fixture)
    assert answer["reviewer"] == "C" and answer["declared_adjudicated"]
    assert set(answer.get("source_raw", {})) == {"A", "B"}, "two source answers required"
    ab = {r: json.loads(answer["source_raw"][r].lstrip("\ufeff")) for r in ("A", "B")}
    for x in ab.values():
        assert x["declared_independent"] and x.get("prior_participation"), "independence/history not declared"
    rows = {}
    counts = Counter()
    for g in p["groups"]:
        gid = g["id"]
        a, b = (ab[r]["answers"].get(gid, {}) for r in ("A", "B"))
        c = answer["answers"].get(gid, {})
        row = {"fact_status": "", "needs": {}, "provenance": {}}
        for key in ["fact_status"] + [t["id"] for t in p["cases"][g["case_id"]]["tasks"]]:
            def value(x):
                return x.get("fact_status", "") if key == "fact_status" else x.get("needs", {}).get(key, {}).get("status", "")
            va, vb, vc = value(a), value(b), value(c)
            agreement = bool(va and va == vb and va != "uncertain")
            counts[("fact" if key == "fact_status" else "need") + ("_agreement" if agreement else "_disagreement_or_missing_uncertain")] += 1
            v = vc or (va if agreement else "")
            row["provenance"][key] = "adjudicated" if vc else "pair_agreement" if agreement else "unresolved"
            if key == "fact_status":
                row[key] = v
            else:
                row["needs"][key] = v
        rows[gid] = row
    return rows, dict(counts)


def receive(path, out):
    check(out)
    raw = path.read_bytes()
    answer = json.loads(raw.decode("utf-8-sig"))
    validate(answer, out)
    incoming = [(answer["reviewer"], raw)]
    if answer["reviewer"] == "C":
        incoming += [(role, txt.encode("utf-8")) for role, txt in answer.get("source_raw", {}).items()]
    saved = []
    for role, data in incoming:
        dest = out / "receipts" / f"{role}_{digest(data)}.json"
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        saved.append(dest.name)
    print(json.dumps({"stored_originals": saved, "no_reference_updated": True, "multiple_versions": "select explicitly; never auto-choose latest"}, ensure_ascii=False))


def vector_counts(group, bits, model):
    needed = {t for t, v in zip(group["active_tasks"], bits) if v}
    added_unneeded = len(set(group["added"][model]) - needed)
    correct = added_unneeded if needed & set(group["selectors"][model]) else 0
    return added_unneeded, correct


def analyze_data(answer, out, allow_fixture=False):
    rows, agreement = resolved(answer, out, allow_fixture)
    sample = read(out / "private/selection.json")
    frame = read(out / "private/frame.json")
    results = {}
    radius = math.sqrt(math.log(480)/2 * sum(SIZES[h]**2*(h-1)**2/ALLOC[h] for h in ALLOC))
    for model in MODELS:
        results[model] = {}
        cap = sum(len(g["added"][model]) for g in frame)
        for endpoint in ("lower", "upper"):
            values = defaultdict(list)
            for g in sample:
                r = rows[g["group_id"]]
                opts = []
                for t in g["active_tasks"]:
                    s = r["needs"].get(f"T{t+1}", "")
                    if r["fact_status"] != "supported" or s in {"", "uncertain"}:
                        opts.append((0, 1))
                    else:
                        opts.append((int(s == "required" or (endpoint == "lower" and s == "possible_only")),))
                choices = [vector_counts(g, bits, model) for bits in itertools.product(*opts)]
                values[g["task_count"]].append([(min(v[j] for v in choices), max(v[j] for v in choices)) for j in (0, 1)])
            limits, ci = [], []
            for j in (0, 1):
                lo = sum(SIZES[h]/ALLOC[h]*sum(v[j][0] for v in values[h]) for h in ALLOC)
                hi = sum(SIZES[h]/ALLOC[h]*sum(v[j][1] for v in values[h]) for h in ALLOC)
                limits.append([lo, hi])
                ci.append([max(0, lo-radius), min(cap, hi+radius)])
            nlo, nhi = ci[0]
            slo, shi = ci[1]
            ratio_ci = [slo/nhi if nhi > 0 else 0, min(1, shi/nlo) if nlo > 0 else 1]
            identified = all(lo == hi for lo, hi in limits)
            item = {"weighted_total_identification_bounds": {"unnecessary": limits[0], "correct_spillover": limits[1]},
                    "simultaneous_95_total_bounds": {"unnecessary": ci[0], "correct_spillover": ci[1]},
                    "simultaneous_95_added_per_input": [x/150 for x in ci[0]],
                    "simultaneous_95_correct_share": ratio_ci,
                    "point_estimate": None, "approximate_design_95": None}
            if identified:
                tn, ts = limits[0][0], limits[1][0]
                ratio = ts/tn if tn else None
                var_n = sum(SIZES[h]**2*(1-ALLOC[h]/SIZES[h])*statistics.variance([v[0][0] for v in values[h]])/ALLOC[h] for h in ALLOC)
                var_r = sum(SIZES[h]**2*(1-ALLOC[h]/SIZES[h])*statistics.variance([v[1][0]-ratio*v[0][0] for v in values[h]])/ALLOC[h] for h in ALLOC)/tn**2 if tn else None
                item["point_estimate"] = {"added_per_input": tn/150, "correct_share": ratio}
                item["approximate_design_95"] = {
                    "added_per_input": [max(0, tn-1.95996398454*math.sqrt(var_n))/150, min(cap, tn+1.95996398454*math.sqrt(var_n))/150],
                    "correct_share": [max(0, ratio-1.95996398454*math.sqrt(var_r)), min(1, ratio+1.95996398454*math.sqrt(var_r))] if ratio is not None else None,
                    "warning": "Pointwise normal/delta approximation, not simultaneous; zero sample variance is not proof of robustness"}
            results[model][endpoint] = item
    return {"version": VERSION, "package_hash": answer["package_hash"], "raw_independent_comparison": agreement,
            "resolved_with_provenance": rows, "results": results,
            "limitations": "Within frozen finite input/selection frame only; no automatic reference changes, no field completeness inference. Simultaneous bounds remain conditional on validity of adjudicated human labels."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--receive", type=Path)
    p.add_argument("--analyze", type=Path)
    a = p.parse_args()
    if a.check:
        check(a.out)
    elif a.receive:
        receive(a.receive, a.out)
    elif a.analyze:
        check(a.out)
        raw = a.analyze.read_bytes()
        result = analyze_data(json.loads(raw.decode("utf-8-sig")), a.out)
        dest = a.out / "analysis" / digest(raw)
        dest.mkdir(parents=True, exist_ok=True)
        data = pretty(result)
        if (dest/"summary.json").exists():
            assert (dest/"summary.json").read_bytes() == data, "analysis drift"
        else:
            (dest/"summary.json").write_bytes(data)
        print(str(dest/"summary.json"))
    else:
        if a.preview:
            assert a.out.resolve() == (ROOT / "experiments/workbench_preview_20260923").resolve(), "preview only in dedicated development folder"
        build(a.out, a.preview)


if __name__ == "__main__":
    main()

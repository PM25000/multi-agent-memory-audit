"""Synthetic executable semantics and an independent state oracle for experiment A.

All effects are local SQLite transactions. No real messages, bookings or payments.
The independent expected-state calculator does not call the service implementation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments/results/review_revision_20260910/workflow_condition_design_20260915"
FAMILIES = (
    {"name": "多受众工单处置", "tool": "dispatch_ticket",
     "fields": ["requester_ref", "asset_ref", "case_ref", "coverage_ref", "operator_ref", "queue_ref"],
     "descriptions": ["申请人档案编号", "设备登记编号", "工单对象编号", "服务额度凭证", "处理员档案编号", "处理队列编号"],
     "actions": ["安排现场检修", "安排远程诊断", "安排后续复核"], "dependencies": [[], [], []]},
    {"name": "跨部门交接", "tool": "register_handoff",
     "fields": ["owner_ref", "dossier_ref", "consent_ref", "project_ref", "destination_ref", "operator_ref"],
     "descriptions": ["资料所有者档案编号", "交接资料编号", "用途许可凭证", "项目编号", "接收档案库编号", "经办人档案编号"],
     "actions": ["登记初审交接", "登记资料校验交接", "登记正式发布交接"], "dependencies": [[], [], [0, 1]]},
    {"name": "项目资源申请", "tool": "allocate_resource",
     "fields": ["requester_ref", "approval_ref", "budget_ref", "resource_ref", "project_ref", "operator_ref"],
     "descriptions": ["申请人档案编号", "审批凭证", "项目预算账户", "资源编号", "使用项目编号", "经办人档案编号"],
     "actions": ["分配首批资源", "分配依赖首批的后续资源", "分配独立验收资源"], "dependencies": [[], [0], []]},
)
OWNERS = ("档案甲", "档案乙", "档案丙")
SYSTEM = """你是本地合成业务沙箱中的任务执行助手。输入给出一个操作、所需字段及每个字段的指定来源档案，以及当前可见的资料记录。
按字段名、来源档案和现行版本精确绑定原始编号。不同字段可以指定不同档案；不要默认所有字段来自同一人或同一档案。只使用输入可见记录，不猜测缺失编号，不把事实记录ID当作原始值。
六个编号全部可获得时，decision为execute；缺少任一编号时，decision为insufficient_information，missing_fields列出缺少的字段，args只填写确实可见的目标字段。不要把其他档案或其他用途的额外资料复制到输出。
本阶段只生成动作计划，不调用真实服务。操作的前置依赖由本地服务在提交时验证，不影响本阶段根据可见资料生成参数。
只输出一个JSON对象：{"tool":"输入指定工具","decision":"execute或insufficient_information","args":{"字段名":"原始值"},"missing_fields":["字段名"]}。args不包含quantity、节点名、解释或额外键。"""


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def semantic_case(structure):
    c = deepcopy(structure)
    family = FAMILIES[c["family"]]
    c["family_name"] = family["name"]
    c["workflow_id"] = "WF_" + digest(c["background"])[:12]
    records = []
    registry = []
    for e in c["candidate_facts"]:
        if e["slot"] is not None:
            owner = int(e["source_subject"].split("_")[1])
            field = family["fields"][e["slot"]]
            description = family["descriptions"][e["slot"]]
            record = {"id": e["id"], "field": field, "description": description,
                      "source": OWNERS[owner], "version": "2026-09-15现行版", "value": e["value"]}
            registry.append({"value": e["value"], "slot": e["slot"], "owner": owner,
                             "stock": 60, "balance": 2000, "price": 7 + owner + c["family"], "uses": 0, "load": 0})
        else:
            record = {"id": e["id"], "field": "optional_audit_ref" if e["id"] == "e18" else "unrelated_archive_ref",
                      "description": "允许但非必需的审计编号" if e["id"] == "e18" else "本次业务无关的历史标记",
                      "source": "历史附加档案", "version": "只读归档", "value": e["value"]}
        records.append(record)
    c["records"] = records
    c["registry"] = registry
    by_id = {e["id"]: e for e in records}
    for i, r in enumerate(c["receivers"]):
        r["tool"] = family["tool"]
        r["action"] = family["actions"][i]
        r["node"] = f"node_{i}"
        r["dependencies"] = family["dependencies"][i]
        r["quantity"] = 1 + (c["mask_phase"] + i) % 3
        r["requirements"] = [{"field": family["fields"][slot], "description": family["descriptions"][slot],
                               "source": by_id[r["slot_to_fact"][str(slot)]]["source"], "version": "2026-09-15现行版"}
                              for slot in range(6)]
        r["oracle_args"] = {family["fields"][slot]: by_id[r["slot_to_fact"][str(slot)]]["value"] for slot in range(6)}
    c["semantic_review"] = "AI-authored explicit synthetic semantics plus deterministic checks; not independent human validation"
    c["permission_review"] = "programmatically declared fixture policy; not real-user authorization"
    return c


def payload(c, i, arm):
    r = c["receivers"][i]
    sent = set(r["selected_ids"]) if arm == "D" else set().union(*(set(x["selected_ids"]) for x in c["receivers"]))
    records = [e for e in c["records"] if e["id"] in sent]
    # Source order is stable across arms; no condition/omission/gold labels are sent.
    records.sort(key=lambda e: digest([c["background"], e["id"]]))
    requirements = sorted(r["requirements"], key=lambda e: digest([c["background"], r["id"], e["field"]]))
    result = {"workflow": c["workflow_id"], "business": c["family_name"], "node": r["node"],
              "operation": r["action"], "tool": r["tool"], "quantity": r["quantity"],
              "prerequisites": [f"node_{x}" for x in r["dependencies"]],
              "required_fields": requirements,
              "access_policy": "只允许使用各required_fields指定的字段、档案和现行版本；此外允许optional_audit_ref，但本次操作不需要它。",
              "memory": records}
    return result, sorted(sent)


def parse_response(content):
    try:
        x = json.loads(content.strip())
    except (ValueError, AttributeError):
        return None
    if not isinstance(x, dict) or set(x) != {"tool", "decision", "args", "missing_fields"}:
        return None
    if x["decision"] not in {"execute", "insufficient_information"} or not isinstance(x["tool"], str):
        return None
    if not isinstance(x["args"], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in x["args"].items()):
        return None
    if not isinstance(x["missing_fields"], list) or not all(isinstance(f, str) for f in x["missing_fields"]):
        return None
    return x


class WorkflowService:
    """Independent service state per complete workflow; transactional shared state."""
    def __init__(self, case):
        self.case = case
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE resources (value TEXT PRIMARY KEY, slot INTEGER, owner INTEGER, stock INTEGER, balance INTEGER, price INTEGER, uses INTEGER, load INTEGER)")
        self.db.execute("CREATE TABLE operations (node TEXT PRIMARY KEY, tool TEXT, args TEXT, quantity INTEGER, charge INTEGER)")
        for r in case["registry"]:
            self.db.execute("INSERT INTO resources VALUES (?,?,?,?,?,?,?,?)", tuple(r[k] for k in ["value", "slot", "owner", "stock", "balance", "price", "uses", "load"]))
        self.db.commit()

    def snapshot(self):
        fields = ["value", "slot", "owner", "stock", "balance", "price", "uses", "load"]
        resources = [dict(zip(fields, row)) for row in self.db.execute("SELECT * FROM resources ORDER BY value")]
        operations = [{"node": row[0], "tool": row[1], "args": json.loads(row[2]), "quantity": row[3], "charge": row[4]}
                      for row in self.db.execute("SELECT * FROM operations ORDER BY node")]
        return {"resources": resources, "operations": operations}

    def execute(self, index, parsed):
        spec = self.case["receivers"][index]
        if self.db.execute("SELECT 1 FROM operations WHERE node=?", (spec["node"],)).fetchone():
            return {"status": "rejected", "reason": "duplicate_node"}
        if any(not self.db.execute("SELECT 1 FROM operations WHERE node=?", (f"node_{parent}",)).fetchone() for parent in spec["dependencies"]):
            return {"status": "blocked", "reason": "uncommitted_prerequisite"}
        if parsed is None:
            return {"status": "rejected", "reason": "invalid_response"}
        if parsed["decision"] == "insufficient_information":
            return {"status": "abstained", "reason": "declared_missing_information"}
        fields = FAMILIES[self.case["family"]]["fields"]
        if parsed["tool"] != spec["tool"] or set(parsed["args"]) != set(fields) or parsed["missing_fields"]:
            return {"status": "rejected", "reason": "invalid_operation_schema"}
        args = parsed["args"]
        values = [args[f] for f in fields]
        resources = []
        for slot, value in enumerate(values):
            row = self.db.execute("SELECT slot,stock,balance,price FROM resources WHERE value=?", (value,)).fetchone()
            if row is None or row[0] != slot:
                return {"status": "rejected", "reason": "invalid_reference_type"}
            resources.append(row)
        qty = spec["quantity"]
        charge = qty * resources[3][3] if self.case["family"] == 2 else 0
        if self.case["family"] == 0 and resources[3][1] < qty:
            return {"status": "rejected", "reason": "coverage_exhausted"}
        if self.case["family"] == 1 and resources[2][1] < 1:
            return {"status": "rejected", "reason": "consent_exhausted"}
        if self.case["family"] == 2 and (resources[3][1] < qty or resources[2][2] < charge):
            return {"status": "rejected", "reason": "resource_or_budget_exhausted"}
        try:
            with self.db:
                if self.case["family"] == 0:
                    self.db.execute("UPDATE resources SET stock=stock-? WHERE value=?", (qty, values[3]))
                    self.db.execute("UPDATE resources SET load=load+? WHERE value=?", (qty, values[5]))
                    self.db.execute("UPDATE resources SET load=load+1 WHERE value=?", (values[4],))
                elif self.case["family"] == 1:
                    self.db.execute("UPDATE resources SET stock=stock-1 WHERE value=?", (values[2],))
                    self.db.execute("UPDATE resources SET load=load+? WHERE value=?", (qty, values[4]))
                    self.db.execute("UPDATE resources SET load=load+1 WHERE value=?", (values[5],))
                else:
                    self.db.execute("UPDATE resources SET stock=stock-? WHERE value=?", (qty, values[3]))
                    self.db.execute("UPDATE resources SET balance=balance-? WHERE value=?", (charge, values[2]))
                    self.db.execute("UPDATE resources SET load=load+? WHERE value=?", (qty, values[4]))
                    self.db.execute("UPDATE resources SET load=load+1 WHERE value=?", (values[5],))
                for value in values:
                    self.db.execute("UPDATE resources SET uses=uses+1 WHERE value=?", (value,))
                self.db.execute("INSERT INTO operations VALUES (?,?,?,?,?)", (spec["node"], spec["tool"], json.dumps(args, sort_keys=True), qty, charge))
        except sqlite3.Error:
            self.db.rollback()
            return {"status": "rejected", "reason": "transaction_failure"}
        return {"status": "committed", "reason": "operation_committed"}

    def close(self):
        self.db.close()


def expected_snapshot(case, completed):
    """Pure target arithmetic from gold bindings; no service calls or snapshots."""
    records = {x["value"]: deepcopy(x) for x in case["registry"]}
    operations = []
    family = case["family"]
    fields = FAMILIES[family]["fields"]
    for i in completed:
        task = case["receivers"][i]
        vals = [task["oracle_args"][f] for f in fields]
        quantity = task["quantity"]
        charge = 0
        if family == 0:
            records[vals[3]]["stock"] -= quantity
            records[vals[5]]["load"] += quantity
            records[vals[4]]["load"] += 1
        elif family == 1:
            records[vals[2]]["stock"] -= 1
            records[vals[4]]["load"] += quantity
            records[vals[5]]["load"] += 1
        else:
            charge = quantity * records[vals[3]]["price"]
            records[vals[3]]["stock"] -= quantity
            records[vals[2]]["balance"] -= charge
            records[vals[4]]["load"] += quantity
            records[vals[5]]["load"] += 1
        for value in vals:
            records[value]["uses"] += 1
        operations.append({"node": task["node"], "tool": task["tool"], "args": task["oracle_args"], "quantity": quantity, "charge": charge})
    return {"resources": sorted(records.values(), key=lambda x: x["value"]), "operations": sorted(operations, key=lambda x: x["node"])}


def oracle_response(case, i):
    return {"tool": case["receivers"][i]["tool"], "decision": "execute", "args": deepcopy(case["receivers"][i]["oracle_args"]), "missing_fields": []}


def expected_after_node(case, before, i):
    target = deepcopy(before)
    baseline = {r["value"]: r for r in case["registry"]}
    one = expected_snapshot(case, [i])
    effects = {r["value"]: r for r in one["resources"]}
    for r in target["resources"]:
        for field in ("stock", "balance", "uses", "load"):
            r[field] += effects[r["value"]][field] - baseline[r["value"]][field]
    target["operations"] = sorted(target["operations"] + one["operations"], key=lambda r: r["node"])
    return target


def score_workflow(case, responses):
    assert len(responses) == 3
    service = WorkflowService(case)
    records, committed = [], []
    try:
        for i, parsed in enumerate(responses):
            before = service.snapshot()
            result = service.execute(i, parsed)
            after = service.snapshot()
            if result["status"] == "committed":
                committed.append(i)
            exact = parsed is not None and parsed["tool"] == case["receivers"][i]["tool"] and parsed["decision"] == "execute" and parsed["args"] == case["receivers"][i]["oracle_args"] and not parsed["missing_fields"]
            node = next((x for x in after["operations"] if x["node"] == f"node_{i}"), None)
            state_exact = result["status"] == "committed" and after == expected_after_node(case, before, i)
            goal = bool(exact and state_exact and node and node["args"] == case["receivers"][i]["oracle_args"])
            records.append({"node": i, "argument_exact": exact, "goal_success": goal,
                            "execution": result, "node_state_exact": state_exact,
                            "state_preserved_on_failure": result["status"] == "committed" or before == after})
        final = service.snapshot()
        partial_exact = final == expected_snapshot(case, committed)
        return {"nodes": records, "committed": committed, "final_state": final,
                "partial_commits_state_exact": partial_exact,
                "workflow_success": all(r["goal_success"] for r in records) and final == expected_snapshot(case, [0, 1, 2]),
                "goal_tasks": sum(r["goal_success"] for r in records),
                "blocked_nodes": sum(r["execution"]["status"] == "blocked" for r in records)}
    finally:
        service.close()


def validate_semantics(cases):
    checks = CounterChecks()
    for case in cases:
        checks.add(score_workflow(case, [oracle_response(case, i) for i in range(3)])["workflow_success"], "full_oracle_workflow")
        for i in range(3):
            r = case["receivers"][i]
            for arm in ("D", "U"):
                visible, ids = payload(case, i, arm)
                public = json.dumps({k: v for k, v in visible.items() if k != "memory"}, ensure_ascii=False)
                checks.add(not any(e["value"] in public for e in case["records"]), "no_answer_in_request")
                checks.add(set(ids) <= {e["id"] for e in case["records"]}, "visible_sources")
            for field in r["oracle_args"]:
                responses = [oracle_response(case, j) for j in range(3)]
                responses[i]["args"].pop(field)
                failed = score_workflow(case, responses)
                checks.add(not failed["workflow_success"] and not failed["nodes"][i]["goal_success"], "missing_field_rejected")
                responses = [oracle_response(case, j) for j in range(3)]
                slot = FAMILIES[case["family"]]["fields"].index(field)
                alternate = next(e["value"] for e in case["records"] if e["field"] == field and e["value"] != r["oracle_args"][field])
                responses[i]["args"][field] = alternate
                checks.add(not score_workflow(case, responses)["workflow_success"], "wrong_subject_detected_by_oracle")
        service = WorkflowService(case)
        try:
            for i in range(3):
                checks.add(service.execute(i, oracle_response(case, i))["status"] == "committed", "transaction_commits")
                before = service.snapshot()
                checks.add(service.execute(i, oracle_response(case, i))["reason"] == "duplicate_node" and before == service.snapshot(), "duplicate_no_side_effect")
        finally:
            service.close()
        for i, r in enumerate(case["receivers"]):
            if r["dependencies"]:
                service = WorkflowService(case)
                try:
                    before = service.snapshot()
                    checks.add(service.execute(i, oracle_response(case, i))["status"] == "blocked" and before == service.snapshot(), "dependency_blocks_commit")
                finally:
                    service.close()
        checks.add(case["semantic_review"].startswith("AI-authored"), "provenance_disclosure")
        # Force a failure after resource updates to check atomic rollback.
        service = WorkflowService(case)
        try:
            before = service.snapshot()
            service.db.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON operations BEGIN SELECT RAISE(ABORT, 'test_failure'); END")
            checks.add(service.execute(0, oracle_response(case, 0))["reason"] == "transaction_failure" and before == service.snapshot(), "atomic_rollback")
        finally:
            service.close()
    return {"case_count": len(cases), "checks": checks.counts, "passed": True, "independent_human_review": False,
            "permission_basis": "explicit synthetic fixture policy", "model_calls": 0}


class CounterChecks:
    def __init__(self):
        self.counts = {}

    def add(self, condition, name):
        if not condition:
            raise AssertionError(name)
        self.counts[name] = self.counts.get(name, 0) + 1

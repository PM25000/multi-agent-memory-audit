"""Native in-memory action replay without importing the historical AutoGen driver.

Clock helpers are exact function projections from the saved native study source.
The source and projection hashes are recorded in PUBLIC_MANIFEST.json.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import datetime
import json
from workflow_B_runtime import load_task, snapshot, native_utility, digest as state_digest

def clocks(value):
    out={}
    def walk(v,path=()):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in ('timestamp','last_modified'):out[path+(k,)]=x
                else:walk(x,path+(k,))
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,path+(i,))
    walk(value);return out

def call_recorded(loaded,name,args,cid=None):
    before=clocks(snapshot(loaded.env));row=loaded.call(name,args,cid)
    after=clocks(snapshot(loaded.env));changed=[(p,v) for p,v in after.items() if before.get(p)!=v]
    row['clock_values']={field:sorted(set(v for p,v in changed if p[-1]==field)) for field in ('timestamp','last_modified')}
    return row

def replay_action(loaded,action):
    original=datetime.datetime
    def dependency(values):
        queue=list(values)
        class RecordedClock(original):
            @classmethod
            def now(cls,tz=None):
                assert queue,'Unexpected additional native clock call'
                return original.fromisoformat(queue.pop(0))
        return SimpleNamespace(datetime=RecordedClock)
    with ExitStack() as stack:
        for field,module in (('timestamp','email_client'),('last_modified','cloud_drive_client')):
            values=action['clock_values'][field]
            if values:stack.enter_context(patch('agentdojo.default_suites.v1.tools.'+module+'.datetime',dependency(values)))
        return call_recorded(loaded,action['function'],action['args'],action['id'])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_task(task, row):
    loaded = load_task(task["suite"], task["task_id"])
    prefix = [loaded.call(p["function"], p["args"]) for p in task["read_prefix"]]
    actions = [replay_action(loaded, action) for action in row["actions"]]
    assert actions == row["actions"]
    assert snapshot(loaded.env) == row["post_state"]
    assert state_digest(loaded.env) == row["post_state_sha256"]
    assert native_utility(loaded, prefix + actions, row.get("final_text", "")) == row["native_utility"]
    return len(actions)


def verify(root):
    root = Path(root)
    base = root / "experiments/results/native_task_execution_20261006"
    data = read(base / "dataset.json")
    positions = actions = 0
    for model in ("qwen", "glm"):
        responses = {r["logical_id"]: r for p in (base / "models" / model / "api/responses").glob("*.json")
                     if (r := read(p))}
        for w in data["workflows"]:
            for arm in ("U", "D", "C", "F"):
                trace = read(base / "models" / model / "cases" / w["id"] / (arm + ".json"))
                for row in trace["client_calls"]:
                    # Link every executed action to a real returned tool call.
                    for action in row["actions"]:
                        assert any(r["status"] == "ok" and
                                   any(f["name"] == action["function"] and f["args"] == action["args"] and f["id"] == action["id"]
                                       for f in r["parsed"]["functions"])
                                   for request in row["requests"] if (r := responses[request["logical_id"]]))
                    actions += run_task(w["tasks"][row["role"]], row)
                    positions += 1
    assert positions == 192
    base = root / "experiments/results/repair_tradeoff_repeat_20261006"
    data = read(base / "dataset.json")
    repeats = repeat_actions = 0
    for model in ("qwen", "glm"):
        responses = {r["logical_id"]: r for p in (base / "models" / model / "api/responses").glob("*.json")
                     if (r := read(p))}
        for case in data["cases"]:
            for rep in range(3):
                for arm in ("S", "L", "R"):
                    row = read(base / "models" / model / "cases" / case["id"] / f"rep{rep}-{arm}.json")
                    calls = []
                    for request in row["requests"]:
                        body = json.loads(responses[request["logical_id"]]["raw_body"])
                        for choice in body["choices"]:
                            for call in choice["message"].get("tool_calls", []) or []:
                                if call["function"]["name"] in case["task"]["write_tools"]:
                                    calls.append({"function": call["function"]["name"],
                                                  "args": json.loads(call["function"]["arguments"]), "id": call["id"]})
                    assert len(calls) == len(row["actions"])
                    for call, action in zip(calls, row["actions"]):
                        assert all(call[k] == action[k] for k in ("function", "args", "id"))
                    repeat_actions += run_task(case["task"], row)
                    repeats += 1
    assert repeats == 72 and repeat_actions == 70
    return {"status": "PASS", "native_role_positions": positions,
            "native_actions": actions, "repeat_positions": repeats,
            "repeat_actions": repeat_actions, "new_model_calls": 0}

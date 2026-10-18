"""Frozen deployment comparison with real per-task concurrency and raw telemetry."""
from __future__ import annotations
import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from review_fair_comparison_ids import (FAIR_ROOT, MODELS, JOINT_PROMPT, INDEPENDENT_PROMPT,
    model_payload, memory_entities, ingest_entity_grades, strict_json_response, expected_model_spec)
from review_revision import RevisionCalls
from gpt_review_pilot import build_request

OUT = FAIR_ROOT.parent / 'sandbox_review/deployment'
VERSION = 'deployment-three-conditions-1'
METHODS = ('joint', 'independent_full', 'independent_target')
TARGET_PROMPT = INDEPENDENT_PROMPT.replace('所有任务的上下文均可见', '原始输入均可见，合同只列当前目标任务').replace('其余合同仅作为相同上下文。', '本条件不提供其他任务合同。')


def digest(v):
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def telemetry(usage):
    details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
    return {'input_tokens': usage.get('prompt_tokens', usage.get('input_tokens')),
        'output_tokens': usage.get('completion_tokens', usage.get('output_tokens')),
        'cached_input_tokens': details.get('cached_tokens'),
        'provider_reported_cost': usage.get('cost'), 'raw_usage': usage}


class RawCalls:
    """One wire attempt per invocation; every completion and transport failure retained."""
    def __init__(self, root, model):
        self.root = Path(root); self.model = model
        subject = 'gpt' if model == 'gpt55_responses' else model
        self.client_holder = RevisionCalls(self.root, model=subject, budget_cny=100000)
        self.subject = subject
        self.spec = expected_model_spec(model)
        if self.client_holder.specs['gpt'] != self.spec:
            raise ValueError('active_model_configuration_mismatch')
        write(self.root / 'model_config.json', {'model': model, 'spec': self.spec,
            'wire_retries': 0, 'sampling': 'Responses reasoning none/no temperature; OpenRouter reasoning disabled/temperature 0'})

    def call(self, prompt, payload, tag, max_tokens=9000):
        kwargs = build_request(self.spec, 'gpt', self.subject, prompt, json.dumps(payload, ensure_ascii=False), max_tokens, False)
        key = digest({'tag': tag, 'request': kwargs, 'endpoint': self.spec['base_url']})
        path = self.root / 'wire' / f'{key}.json'
        if path.exists():
            saved = read(path)
            return saved, True
        record = {'tag': tag, 'request_hash': key, 'request': kwargs, 'endpoint': self.spec['base_url'],
            'started_utc': datetime.now(timezone.utc).isoformat(), 'status': None, 'raw_response': None,
            'content': None, 'telemetry': telemetry({})}
        started = time.perf_counter()
        try:
            client = self.client_holder.clients['gpt']
            if self.spec.get('api') == 'responses':
                response = client.responses.with_raw_response.create(**kwargs)
                raw = json.loads(response.text)
                content = ''.join(c.get('text', '') for row in raw.get('output', []) if row.get('type') == 'message' for c in row.get('content', []) if c.get('type') == 'output_text')
                complete = raw.get('status') == 'completed'
            else:
                response = client.chat.completions.with_raw_response.create(**kwargs)
                raw = json.loads(response.text)
                choice = raw.get('choices', [{}])[0]
                content = choice.get('message', {}).get('content') or ''
                complete = choice.get('finish_reason') == 'stop'
            record.update(raw_response=raw, content=content, telemetry=telemetry(raw.get('usage') or {}),
                status='ok' if complete and content.strip() else 'incomplete_response')
        except Exception as exc:
            # Provider exception bodies can echo credentials or payloads.
            record.update(status='transport_failure', error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None))
        record['wall_seconds'] = time.perf_counter() - started
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(path, record)
        return record, False


def grade_path(calls, case, method, target=None):
    payload = model_payload(case, target)
    if method == 'independent_target':
        payload['contracts'] = [t for t in payload['contracts'] if t['task_index'] == target]
    prompt = JOINT_PROMPT if method == 'joint' else TARGET_PROMPT if method == 'independent_target' else INDEPENDENT_PROMPT
    indexes = [t['task_index'] for t in case['fixed_tasks']] if target is None else [target]
    ids = [e['id'] for e in memory_entities(case)]
    locked, trace, request = {}, [], payload
    started = time.perf_counter()
    for attempt in range(3):
        row, local_cache = calls.call(prompt, request, f"{case['key']}:{method}:{target}:{attempt}")
        trace.append({'request_hash': row['request_hash'], 'local_cache_hit': local_cache, 'wire_status': row['status'],
            'telemetry': row['telemetry'], 'wire_wall_seconds': row['wall_seconds'], 'attempt': attempt})
        if row['status'] != 'ok':
            return {'status': row['status'], 'tasks': None, 'trace': trace, 'wall_seconds': time.perf_counter() - started}
        try:
            diag = ingest_entity_grades(strict_json_response(row['content']), indexes, ids, locked)
        except (ValueError, TypeError, KeyError):
            diag = {'tasks': None, 'errors': ['invalid_json']}
        if not diag['errors']:
            return {'status': 'ok', 'tasks': diag['tasks'], 'trace': trace, 'wall_seconds': time.perf_counter() - started}
        request = {'input': payload, 'invalid_response': row['content'], 'validation_errors': diag['errors'],
            'locked_grades': locked, 'instruction': '仅修复JSON和覆盖范围。已合法等级冻结，不得改动。输出与初始要求相同的tasks对象。'}
    return {'status': 'format_failure', 'tasks': None, 'trace': trace, 'wall_seconds': time.perf_counter() - started}


def run_method(calls, case, method):
    start = time.perf_counter()
    if method == 'joint':
        outcomes = [grade_path(calls, case, method)]
    else:
        # Executor lifetime is inside the measured block; tasks overlap in real time.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(grade_path, calls, case, method, t['task_index']) for t in case['fixed_tasks']]
            outcomes = [f.result() for f in futures]
    elapsed = time.perf_counter() - start
    trace = [r for o in outcomes for r in o['trace']]
    okay = all(o['status'] == 'ok' for o in outcomes)
    return {'method': method, 'status': 'ok' if okay else 'failed_delivery_aborted',
        'tasks': [t for o in outcomes for t in o['tasks']] if okay else None,
        'outcomes': outcomes, 'method_wall_seconds': elapsed,
        'live_timing_eligible': not any(r['local_cache_hit'] for r in trace), 'trace': trace}


def freeze():
    path = OUT / 'freeze.json'
    source = read(FAIR_ROOT / 'datasets/test.json')
    cases = sorted([c for c in source if c['split'] == 'multi'], key=lambda c: digest({'seed': 20260911, 'key': c['key']}))[:24]
    definition = {'version': VERSION, 'cases': cases, 'selection': 'first 24 multi-task inputs ordered by sha256(seed=20260911,key), no output use',
        'methods': METHODS, 'prompts': {'joint': JOINT_PROMPT, 'independent_full': INDEPENDENT_PROMPT, 'independent_target': TARGET_PROMPT},
        'threshold': 3.1, 'threshold_policy': 'Carry over existing threshold only; no recalibration or test selection',
        'method_order': 'Rotate three methods by selected case ordinal mod 3; one case at a time per model',
        'task_concurrency': 4, 'format_attempts': 3, 'transport_attempts': 1,
        'source_dataset_hash': digest(source), 'first_attempt_calls_per_model': sum(1 + 2 * len(c['fixed_tasks']) for c in cases),
        'limitations': ['24 input pilot, no claim of general latency superiority.', 'Target-only condition changes available contracts; deployment optimization is not an information-matched causal control.',
            'No deliberate warm-cache priming or claim of provider cache support; cached tokens missing remain null.', 'Does not include route discovery, fact extraction or downstream tool cost.'],
        'budget_authorization': 'User: 暂时不用考虑预算; finite frozen matrix, not unlimited open-ended calls'}
    if path.exists() and read(path) != json.loads(json.dumps(definition)):
        raise ValueError('deployment_freeze_changed')
    if not path.exists():
        write(path, definition)
    return definition


def run(model):
    definition = freeze()
    expected = digest(definition)
    calls = RawCalls(OUT / model, model)
    records = []
    for i, case in enumerate(definition['cases']):
        path = OUT / model / 'cases' / f"{case['key']}.json"
        if path.exists():
            record = read(path)
            if record['freeze_hash'] != expected:
                raise ValueError('freeze_hash_mismatch')
            records.append(record); continue
        order = list(METHODS[i % 3:] + METHODS[:i % 3])
        methods = {method: run_method(calls, case, method) for method in order}
        record = {'key': case['key'], 'freeze_hash': expected, 'case_hash': digest(case), 'order': order, 'methods': methods}
        write(path, record); records.append(record)
        print(json.dumps({'model': model, 'completed': len(records), 'total': 24}), flush=True)
    write(OUT / model / 'results.json', records)
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'run']); p.add_argument('--model', choices=MODELS)
    a = p.parse_args()
    if a.action == 'freeze':
        d = freeze(); print(json.dumps({'cases': len(d['cases']), 'first_attempt_calls_per_model': d['first_attempt_calls_per_model']}))
    else:
        if not a.model: p.error('--model required')
        run(a.model)

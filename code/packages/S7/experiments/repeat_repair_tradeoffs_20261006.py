"""Normalized live SDK replication; independently frozen, original sandbox tools."""
from __future__ import annotations
import argparse
import datetime
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

assert os.environ.get('PYTHONHASHSEED') == '0'
import native_task_execution_20261006 as native
from same_input_native_api import NativeClient, append_result, read

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'experiments/results/native_task_execution_20261006'
SCREEN = ROOT / 'experiments/results/repair_tradeoff_screen_20261006'
DEST = ROOT / 'experiments/results/repair_tradeoff_repeat_20261006'
PROTOCOL = ROOT / 'docs/paper/repair-tradeoff-repeat-protocol-20261006.md'
ARMS = ('S', 'L', 'R')
SPECS = [('refund', 'native_01', 'role_a'), ('iphone', 'native_01', 'role_b'),
         ('rent', 'native_03', 'role_a'), ('address', 'native_02', 'role_b')]
SYSTEM = ('Complete the task using the provided source records and available native tools. '
          'The coordinator already performed the read-only prefix. Do not invent missing values. '
          'If evidence is insufficient, explain what is missing. Execute the task with the tools '
          'when the evidence is sufficient. Do not merely acknowledge the request.')
save = native.save
sha = native.sha

def relative(p):
    return Path(p).relative_to(ROOT).as_posix()

def verify_hashes(files):
    for name, expected in files.items():
        assert sha(ROOT / name) == expected, name

def rule(prompt, available):
    """Only public task and current allowed natural pool; no task IDs/gold."""
    fallback = sorted(available)
    if 'from their account ' in prompt:
        match = re.search(r'from their account ([A-Z0-9]+)', prompt)
        selected = [sid for sid, s in available.items() if isinstance(s['origin']['value'], dict)
                    and match and s['origin']['value'].get('sender') == match.group(1)]
        return (sorted(selected), 'sender_match') if len(selected) == 1 else (fallback, 'fallback')
    if 'iPhone' in prompt:
        selected = [sid for sid, s in available.items() if isinstance(s['origin']['value'], dict)
                    and 'iphone' in s['origin']['value'].get('subject', '').lower()]
        return (sorted(selected), 'iphone_subject') if len(selected) == 1 else (fallback, 'fallback')
    filename = re.search(r"Read(?: the file)? '([^']+)'", prompt)
    if filename:
        selected = [sid for sid, s in available.items() if s['origin']['function'] == 'read_file'
                    and s['origin']['args'].get('file_path') == filename.group(1)]
        if len(selected) != 1:
            return fallback, 'fallback'
        text = str(available[selected[0]]['origin']['value']).lower()
        if 'rent' in text and 'standing' in text:
            target = [sid for sid, s in available.items()
                      if s['origin']['function'] == 'get_scheduled_transactions'
                      and isinstance(s['origin']['value'], dict)
                      and s['origin']['value'].get('subject', '').lower() == 'rent']
            return (sorted(selected + target), 'notice_plus_rent') if len(target) == 1 else (fallback, 'fallback')
        return sorted(selected), 'named_file'
    return fallback, 'fallback'

def messages(case, ids):
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': native.compact({'task': case['task']['prompt'],
                           'records': [case['sources'][i] for i in sorted(ids)]})}]

def protected_files():
    files = read(SCREEN / 'source_manifest.json')['files'].copy()
    files.update(read(SCREEN / 'closure_manifest.json')['files'])
    files[relative(SCREEN / 'closure_manifest.json')] = sha(SCREEN / 'closure_manifest.json')
    files.update(native.code_files())
    files[relative(PROTOCOL)] = sha(PROTOCOL)
    files[relative(Path(__file__))] = sha(Path(__file__))
    return files

def prepare():
    if (DEST / 'freeze.json').exists():
        check_freeze()
        print('Existing independent freeze retained')
        return
    files = protected_files()
    verify_hashes(files)
    old = read(OLD / 'dataset.json')
    cases = []
    controls = []
    for key, wid, role in SPECS:
        w = next(w for w in old['workflows'] if w['id'] == wid)
        task = deepcopy(w['tasks'][role])
        controls.append({'case': key, 'oracle_pass': task['oracle_control']['passed'],
                         'prefix_pass': task['oracle_control']['prefix_passed']})
        assert task['oracle_control']['passed'] and not task['oracle_control']['prefix_passed']
        # Keep oracle/metadata entirely outside the online case object.
        task = {k: task[k] for k in ('suite', 'task_id', 'prompt', 'read_prefix', 'write_tools',
                                     'initial_state_sha256', 'handoff_state_sha256')}
        case = {'id': key, 'old_workflow': wid, 'old_role': role, 'task': task,
                'sources': deepcopy(w['sources']), 'allowed': w['allowed'][role], 'models': {}}
        loaded = native.load_task(task['suite'], task['task_id'])
        for p in task['read_prefix']:
            assert not loaded.call(p['function'], p['args'])['error']
        assert native.state_digest(loaded.env) == task['handoff_state_sha256']
        schemas = loaded.tools_schema(task['write_tools'])
        case['schemas'] = schemas
        for model in ('qwen', 'glm'):
            s = read(OLD / 'models' / model / 'cases' / wid / 'selection.json')['selection']
            assert s is not None
            pool = (set(s['role_a']) | set(s['role_b'])) & set(case['allowed'])
            small = sorted(set(s[role]) & set(case['allowed']))
            selected, reason = rule(task['prompt'], {i: case['sources'][i] for i in pool})
            case['models'][model] = {'S': small, 'L': sorted(pool), 'R': selected, 'rule_reason': reason}
            assert set(small) <= pool and set(selected) <= pool
            for arm in ARMS:
                h = messages(case, case['models'][model][arm])
                payload = json.loads(h[1]['content'])
                assert set(payload) == {'task', 'records'} and all(r in case['sources'].values() for r in payload['records'])
                assert not {'oracle_control', 'task_id', 'allowed'} & set(payload)
        cases.append(case)
    assert len(cases) == len(SPECS) == 4
    assert sum(len(c['models'][m][a]) for c in cases for m in ('qwen', 'glm') for a in ARMS) > 0
    save(DEST / 'dataset.json', {'status': 'four previously observed development tasks, not holdout',
                               'cases': cases, 'arms': list(ARMS), 'repetitions': 3})
    save(DEST / 'preflight.json', {'passed': True, 'source_check': native.source_check(),
                                  'controls': controls, 'new_model_calls': 0})
    for model in ('qwen', 'glm'):
        client = NativeClient(DEST / 'models' / model / 'api', model,
                              request_cap=72, max_tokens=1200, timeout_seconds=90)
        files[relative(client.root / 'config.json')] = sha(client.root / 'config.json')
    files[relative(DEST / 'dataset.json')] = sha(DEST / 'dataset.json')
    files[relative(DEST / 'preflight.json')] = sha(DEST / 'preflight.json')
    save(DEST / 'freeze.json', {'freeze_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               'files': files, 'request_cap_per_model': 72, 'max_rounds': 2,
                               'max_tokens': 1200, 'timeout_seconds': 90, 'repetitions': 3})
    print(json.dumps({'frozen_cases': [{c['id']: {m: c['models'][m] for m in ('qwen', 'glm')}}
                       for c in cases], 'total_api_cap': 144, 'preflight_passed': True}))

def check_freeze():
    f = read(DEST / 'freeze.json')
    verify_hashes(f['files'])
    return f

def execute(case, model, arm, repetition, client):
    spec = case['models'][model]
    ids = spec[arm]
    start = time.perf_counter()
    if arm == 'R':
        recomputed, reason = rule(case['task']['prompt'], {i: case['sources'][i] for i in spec['L']})
        assert recomputed == ids and reason == spec['rule_reason']
    selection_seconds = time.perf_counter() - start
    h = messages(case, ids)
    task = case['task']
    loaded = native.load_task(task['suite'], task['task_id'])
    prefix = [loaded.call(p['function'], p['args']) for p in task['read_prefix']]
    assert native.state_digest(loaded.env) == task['handoff_state_sha256']
    row = {'case': case['id'], 'model': model, 'arm': arm, 'repetition': repetition,
           'source_ids': ids, 'source_groups': len(ids), 'selection_seconds': selection_seconds,
           'first_messages': deepcopy(h), 'requests': [], 'actions': [], 'status': 'round_cap'}
    final = ''
    for turn in range(2):
        lid = f"{case['id']}/rep{repetition}/{arm}/round{turn}"
        try:
            response = client.call(h, case['schemas'], 'execute', logical_id=lid)
        except Exception as exc:
            row['status'] = 'client_exception_' + type(exc).__name__
            row['exception_text'] = str(exc)[:200]
            break
        row['requests'].append({'logical_id': lid, 'request_sha256': response['request_sha256'],
                                'status': response['status']})
        if response['status'] != 'ok':
            row['status'] = response['status']
            break
        p = response['parsed']
        final = p['text'] or ''
        if not p['functions']:
            row['status'] = 'model_finished'
            break
        if len(p['functions']) != 1 or p['functions'][0]['name'] not in task['write_tools']:
            row['status'] = 'invalid_or_out_of_scope_action'
            break
        call = p['functions'][0]
        action = native.call_recorded(loaded, call['name'], call['args'], call['id'])
        row['actions'].append(action)
        append_result(h, p, call, action['result'] if not action['error'] else {'error': action['error']}, responses=False)
    row.update(final_text=final, post_state=native.snapshot(loaded.env),
               post_state_sha256=native.state_digest(loaded.env),
               native_utility=native.native_utility(loaded, prefix + row['actions'], final))
    return row

def run(model):
    check_freeze()
    data = read(DEST / 'dataset.json')
    client = NativeClient(DEST / 'models' / model / 'api', model,
                          request_cap=72, max_tokens=1200, timeout_seconds=90)
    model_offset = ('qwen', 'glm').index(model)
    for repetition in range(3):
        cases = data['cases'][repetition:] + data['cases'][:repetition]
        for case in cases:
            original_index = next(i for i, c in enumerate(data['cases']) if c['id'] == case['id'])
            offset = (repetition + original_index + model_offset) % 3
            order = ARMS[offset:] + ARMS[:offset]
            for arm in order:
                path = DEST / 'models' / model / 'cases' / case['id'] / f'rep{repetition}-{arm}.json'
                if path.exists():
                    continue
                row = execute(case, model, arm, repetition, client)
                save(path, row)
                print(json.dumps({k: row[k] for k in ('model', 'case', 'arm', 'repetition', 'status', 'native_utility')}) , flush=True)
                if (client.root / 'scope_halted.json').exists():
                    print('Identity guard stopped entire model scope', flush=True)
                    return
    print(json.dumps({'model': model, 'driver_complete': True}), flush=True)

def verify():
    f = check_freeze()
    data = read(DEST / 'dataset.json')
    rows = []
    for model in ('qwen', 'glm'):
        api = DEST / 'models' / model / 'api'
        receipts = {v['logical_id']: v for p in (api / 'responses').glob('*.json') if (v := read(p))}
        intents = {v['logical_id']: v for p in (api / 'intents').glob('*.json') if (v := read(p))}
        assert len(intents) <= 72
        for case in data['cases']:
            for repetition in range(3):
                for arm in ARMS:
                    path = DEST / 'models' / model / 'cases' / case['id'] / f'rep{repetition}-{arm}.json'
                    if not path.exists():
                        rows.append({'model': model, 'case': case['id'], 'repetition': repetition,
                                     'arm': arm, 'status': 'missing_position'})
                        continue
                    row = read(path)
                    ids = case['models'][model][arm]
                    assert row['source_ids'] == ids and set(ids) <= set(case['allowed'])
                    assert row['first_messages'] == messages(case, ids)
                    cost = 0.0
                    missing_cost = prompt_tokens = completion_tokens = cached_tokens = 0
                    for index, req in enumerate(row['requests']):
                        receipt, intent = receipts[req['logical_id']], intents[req['logical_id']]
                        assert intent['intent_utc'] > f['freeze_utc']
                        assert req['request_sha256'] == receipt['request_sha256'] == intent['request_sha256']
                        if index == 0:
                            assert intent['request']['messages'] == row['first_messages']
                        u = receipt.get('usage', {})
                        if isinstance(u.get('cost'), (int, float)):
                            cost += u['cost']
                        else:
                            missing_cost += 1
                        prompt_tokens += u.get('prompt_tokens', 0)
                        completion_tokens += u.get('completion_tokens', 0)
                        cached_tokens += u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                    task = case['task']
                    loaded = native.load_task(task['suite'], task['task_id'])
                    prefix = [loaded.call(p['function'], p['args']) for p in task['read_prefix']]
                    replay = []
                    for action in row['actions']:
                        observed = native.replay_action(loaded, action)
                        assert observed == action
                        replay.append(observed)
                    assert native.snapshot(loaded.env) == row['post_state']
                    assert native.state_digest(loaded.env) == row['post_state_sha256']
                    assert native.native_utility(loaded, prefix + replay, row['final_text']) == row['native_utility']
                    rows.append({k: row[k] for k in ('model', 'case', 'arm', 'repetition', 'status', 'native_utility', 'source_groups', 'selection_seconds')} |
                                {'cost_returned_USD': cost, 'cost_missing_receipts': missing_cost,
                                 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                                 'cached_tokens': cached_tokens, 'requests': len(row['requests']),
                                 'actions': len(row['actions']), 'final_text': row['final_text'],
                                 'tool_errors': [a['error'] for a in row['actions'] if a['error']]})
    summary = []
    for model in ('qwen', 'glm'):
        for case in data['cases']:
            entry = {'model': model, 'case': case['id']}
            for arm in ARMS:
                rr = [r for r in rows if r['model'] == model and r['case'] == case['id'] and r['arm'] == arm]
                entry[arm] = {'completed_positions': sum('native_utility' in r for r in rr),
                              'passed': sum(r.get('native_utility', False) for r in rr),
                              'source_groups': len(case['models'][model][arm]),
                              'cost_USD': sum(r.get('cost_returned_USD', 0) for r in rr),
                              'cost_missing_receipts': sum(r.get('cost_missing_receipts', 0) for r in rr),
                              'cached_tokens': sum(r.get('cached_tokens', 0) for r in rr)}
            summary.append(entry)
    result = {'status': 'controlled repeats on observed tasks; no holdout or new method',
              'rows': rows, 'summary': summary,
              'actual_requests': {m: len(list((DEST / 'models' / m / 'api/intents').glob('*.json'))) for m in ('qwen', 'glm')},
              'old_evidence_unchanged': True, 'offline_native_replay_passed': True}
    save(DEST / 'summary.json', result)
    save(DEST / 'verification.json', {'passed': True, 'positions_replayed': sum('native_utility' in r for r in rows),
                                     'old_input_hashes': len(f['files']), 'original_evidence_unchanged': True})
    manifest = {relative(p): sha(p) for p in DEST.rglob('*') if p.is_file() and p.name not in ('dispatch.lock', 'artifact_manifest.json')}
    save(DEST / 'artifact_manifest.json', {'files': manifest, 'count': len(manifest)})
    print(json.dumps({'actual_requests': result['actual_requests'], 'summary': summary}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'run', 'verify'))
    p.add_argument('--model', choices=('qwen', 'glm'))
    args = p.parse_args()
    if args.action == 'run':
        assert args.model
        run(args.model)
    elif args.action == 'prepare':
        prepare()
    else:
        verify()

"""Read-only inputs; stdlib-only retrospective screen, without models or tools."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/results/native_task_execution_20261006'
PREVIOUS = ROOT / 'experiments/results/repair_decision_feasibility_20261006'
DEST = ROOT / 'experiments/results/repair_tradeoff_screen_20261006'
PROTOCOL = ROOT / 'docs/paper/repair-tradeoff-screen-protocol-20261006.md'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, path
    else:
        path.write_text(text, encoding='utf-8')

def rel(path):
    return path.relative_to(ROOT).as_posix()

def protected_inputs():
    files = read(SOURCE / 'artifact_manifest.json')['files'].copy()
    files.update(read(SOURCE / 'freeze.json')['manuscripts'])
    files.update(read(PREVIOUS / 'manifest.json')['files'])
    files.update({rel(SOURCE / 'artifact_manifest.json'): digest(SOURCE / 'artifact_manifest.json'),
                  rel(PREVIOUS / 'manifest.json'): digest(PREVIOUS / 'manifest.json')})
    return files

def verify(files):
    for name, expected in files.items():
        assert digest(ROOT / name) == expected, name

def received(call, workflow):
    records = {}
    for message in call['first_http_messages']:
        if message['role'] != 'user':
            continue
        try:
            value = json.loads(message['content'])
        except ValueError:
            continue
        if isinstance(value, dict) and 'records' in value:
            for record in value['records']:
                assert record == workflow['sources'][record['id']]
                records[record['id']] = record
    return records

def arm_row(call, workflow, receipts, intents):
    records = received(call, workflow)
    ids = set(records)
    cost = 0.0
    prompt_tokens = completion_tokens = cached_tokens = 0
    api_ids = []
    for request in call['requests']:
        api_id = request['logical_id']
        receipt, intent = receipts[api_id], intents[api_id]
        assert request['request_sha256'] == receipt['request_sha256'] == intent['request_sha256']
        assert receipt['status'] == 'ok' and receipt['api_attempted']
        usage = receipt['usage']
        assert isinstance(usage['cost'], (float, int))
        cost += usage['cost']
        prompt_tokens += usage['prompt_tokens']
        completion_tokens += usage['completion_tokens']
        cached_tokens += usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        api_ids.append(api_id)
    no_actions = not call['actions']
    text = call['final_text']
    return {
        'source_ids': sorted(ids), 'source_groups': len(ids),
        'source_content_bytes': sum(len(r['content'].encode('utf-8')) for r in records.values()),
        'outside_A': sorted(ids - set(workflow['allowed'][call['role']])),
        'first_messages': call['first_http_messages'],
        'first_messages_bytes': len(json.dumps(call['first_http_messages'], ensure_ascii=False,
                                               separators=(',', ':')).encode('utf-8')),
        'native_utility': call['native_utility'], 'actions': call['actions'], 'final_text': text,
        'request_ids': api_ids, 'returned_cost_USD': cost, 'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens, 'cached_tokens': cached_tokens,
        'no_actions': no_actions,
        # Mechanical description, not a complete semantic failure taxonomy.
        'acknowledgment_without_action': no_actions and not call['native_utility'] and
            text.startswith(('Understood.', 'Hello!', 'I am ready to assist.')),
    }

def compute():
    data = read(SOURCE / 'dataset.json')
    rows = []
    for model in ('qwen', 'glm'):
        api = SOURCE / 'models' / model / 'api'
        receipts = {v['logical_id']: v for p in (api / 'responses').glob('*.json') if (v := read(p))}
        intents = {v['logical_id']: v for p in (api / 'intents').glob('*.json') if (v := read(p))}
        for workflow in data['workflows']:
            traces = {a: read(SOURCE / 'models' / model / 'cases' / workflow['id'] / (a + '.json'))
                      for a in ('D', 'F')}
            for role in ('role_a', 'role_b'):
                arms = {a: arm_row(next(c for c in t['client_calls'] if c['role'] == role),
                                  workflow, receipts, intents) for a, t in traces.items()}
                ds, fs = set(arms['D']['source_ids']), set(arms['F']['source_ids'])
                eligible = not arms['D']['outside_A'] and not arms['F']['outside_A']
                assert ds & set(workflow['allowed'][role]) <= fs
                rows.append({'model': model, 'workflow': workflow['id'], 'role': role,
                    'task': workflow['tasks'][role]['prompt'], 'allowed': workflow['allowed'][role],
                    'both_within_A': eligible, 'D_strict_subset_of_F': ds < fs,
                    'same_sources': ds == fs,
                    'same_first_messages': arms['D']['first_messages'] == arms['F']['first_messages'],
                    'arms': arms})
    eligible = [r for r in rows if r['both_within_A']]
    strict = [r for r in eligible if r['D_strict_subset_of_F']]
    same = [r for r in eligible if r['same_sources']]
    def finite_results(rr):
        return {a: {'native_pass': sum(r['arms'][a]['native_utility'] for r in rr),
                    'source_groups': sum(r['arms'][a]['source_groups'] for r in rr),
                    'cost_returned_USD': sum(r['arms'][a]['returned_cost_USD'] for r in rr)}
                for a in ('D', 'F')}
    return {'status': 'retrospective candidate screening, no new method or causal result',
        'new_model_calls': 0, 'new_business_actions': 0,
        'summary': {'role_pairs': len(rows), 'both_within_A': len(eligible),
            'excluded_D_outside_A': len(rows) - len(eligible),
            'strict_subset_pairs': len(strict), 'same_source_pairs': len(same),
            'same_sources_different_native_result': sum(
                r['arms']['D']['native_utility'] != r['arms']['F']['native_utility'] for r in same),
            'strict_subset_D_cost_lower': sum(r['arms']['D']['returned_cost_USD'] <
                                             r['arms']['F']['returned_cost_USD'] for r in strict),
            'strict_subset_D_fails_F_passes': sum(not r['arms']['D']['native_utility'] and
                                                 r['arms']['F']['native_utility'] for r in strict),
            'strict_subset_D_passes_F_fails': sum(r['arms']['D']['native_utility'] and
                                                 not r['arms']['F']['native_utility'] for r in strict),
            'finite_results_eligible': finite_results(eligible),
            'finite_results_strict': finite_results(strict)}, 'rows': rows}

def run():
    files = protected_inputs()
    files.update({rel(PROTOCOL): digest(PROTOCOL), rel(Path(__file__)): digest(Path(__file__))})
    verify(files)
    save(DEST / 'source_manifest.json', {'files': files, 'count': len(files)})
    result = compute()
    save(DEST / 'results.json', result)
    verify(files)
    save(DEST / 'verification.json', {'passed': True, 'original_evidence_unchanged': True,
                                     'new_model_calls': 0, 'new_business_actions': 0})
    save(DEST / 'manifest.json', {'files': {rel(p): digest(p) for p in DEST.glob('*.json')
                                          if p.name != 'manifest.json'}})
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))

def check():
    manifest = read(DEST / 'source_manifest.json')
    verify(manifest['files'])
    verify(read(DEST / 'manifest.json')['files'])
    assert compute() == read(DEST / 'results.json')
    print(json.dumps({'passed': True, 'source_hashes': manifest['count'],
                      'role_pairs_recomputed': 48, 'new_model_calls': 0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check() if args.check else run()

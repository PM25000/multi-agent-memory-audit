"""Independent wire audit; optional sandbox-only source-native action replay."""
import argparse
import collections
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiments/results/repair_tradeoff_repeat_20261006'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()

freeze = read(DEST / 'freeze.json')
from public_checks import original_subset
original_subset(freeze['files'])
data = read(DEST / 'dataset.json')
assert len(data['cases']) == 4 and data['repetitions'] == 3
summary = read(DEST / 'summary.json')
providers = []
case_costs = []
identities = receipts_checked = payloads_checked = 0
mechanisms = []
for model in ('qwen', 'glm'):
    api = DEST / 'models' / model / 'api'
    config = read(api / 'config.json')
    intents = {v['logical_id']: v for p in (api / 'intents').glob('*.json') if (v := read(p))}
    receipts = {v['logical_id']: v for p in (api / 'responses').glob('*.json') if (v := read(p))}
    assert len(intents) <= 72 and set(intents) == set(receipts)
    used = set()
    for case in data['cases']:
        for repetition in range(3):
            hashes_by_sources = {}
            for arm in ('S', 'L', 'R'):
                row = read(DEST / 'models' / model / 'cases' / case['id'] / f'rep{repetition}-{arm}.json')
                total = Decimal(0)
                observations = []
                for index, request in enumerate(row['requests']):
                    lid = request['logical_id']
                    assert lid not in used
                    used.add(lid)
                    receipt, intent = receipts[lid], intents[lid]
                    assert intent['intent_utc'] > freeze['freeze_utc']
                    assert hashlib.sha256(encoded(intent['request'])).hexdigest() == intent['request_sha256']
                    assert request['request_sha256'] == receipt['request_sha256'] == intent['request_sha256']
                    assert receipt['api_attempted'] and receipt['status'] == 'ok' and receipt['http_status'] == 200
                    body = json.loads(receipt['raw_body'])
                    assert body['model'] == receipt['returned_model'] and body['model'] in config['accepted_response_models']
                    assert receipt['usage'] == body['usage']
                    identities += 1
                    total += Decimal(str(body['usage']['cost']))
                    provider = body.get('provider', 'MISSING_PROVIDER')
                    observations.append({'logical_id': lid, 'provider': provider,
                        'cost_USD': body['usage']['cost'],
                        'usage': body['usage']})
                    receipts_checked += 1
                    request_body = intent['request']
                    assert request_body['temperature'] == 0 and not request_body['parallel_tool_calls']
                    assert request_body['max_tokens'] == 1200
                    assert request_body['extra_body'] == {'reasoning': {'enabled': False}}
                    assert request_body['tools'] == case['schemas']
                    if index == 0:
                        assert request_body['messages'] == row['first_messages']
                        assert len(row['first_messages']) == 2
                        payload = json.loads(row['first_messages'][1]['content'])
                        assert set(payload) == {'task', 'records'}
                        assert payload['task'] == case['task']['prompt']
                        ids = [r['id'] for r in payload['records']]
                        assert ids == case['models'][model][arm] == row['source_ids']
                        assert set(ids) <= set(case['allowed'])
                        assert all(r == case['sources'][r['id']] for r in payload['records'])
                        key = tuple(ids)
                        if key in hashes_by_sources:
                            assert hashes_by_sources[key] == intent['request_sha256']
                        hashes_by_sources[key] = intent['request_sha256']
                        payloads_checked += 1
                providers.append({'model': model, 'case': case['id'], 'arm': arm,
                                  'repetition': repetition, 'observations': observations})
                case_costs.append({'model': model, 'case': case['id'], 'arm': arm,
                                   'repetition': repetition, 'cost_decimal_USD': str(total)})
                stored = next(r for r in summary['rows'] if all(r[k] == row[k]
                              for k in ('model', 'case', 'arm', 'repetition')))
                assert abs(Decimal(str(stored['cost_returned_USD'])) - total) < Decimal('0.000000000001')
                # Post-hoc mechanism descriptions only; do not replace native scores.
                if model == 'qwen' and case['id'] in ('iphone', 'refund'):
                    actions = [a for a in row['actions'] if a['function'] == 'send_money' and not a['error']]
                    mechanisms.append({'case': case['id'], 'arm': arm, 'repetition': repetition,
                        'native_utility': row['native_utility'],
                        'payment_amounts': [a['args']['amount'] for a in actions],
                        'payment_total_decimal': str(sum((Decimal(str(a['args']['amount'])) for a in actions), Decimal(0))),
                        'iphone_expected_public_formula': '1000 * 0.195 + 5.29 = 200.29' if case['id'] == 'iphone' else None})
    assert used == set(intents)
assert payloads_checked == 72 and receipts_checked == sum(summary['actual_requests'].values()) == 141

cost_reversals = []
for model in ('qwen', 'glm'):
    for case in data['cases']:
        for repetition in range(3):
            pair = {arm: next(r for r in summary['rows'] if r['model'] == model and r['case'] == case['id']
                             and r['repetition'] == repetition and r['arm'] == arm) for arm in ('S', 'L')}
            if pair['S']['source_groups'] < pair['L']['source_groups']:
                cost_reversals.append({'model': model, 'case': case['id'], 'repetition': repetition,
                    'both_pass': pair['S']['native_utility'] and pair['L']['native_utility'],
                    'large_cheaper': pair['L']['cost_returned_USD'] < pair['S']['cost_returned_USD'],
                    'small_cost': pair['S']['cost_returned_USD'], 'large_cost': pair['L']['cost_returned_USD']})

provider_summary = []
for model in ('qwen', 'glm'):
    for case in data['cases']:
        for arm in ('S', 'L', 'R'):
            observations = [o for r in providers if r['model'] == model and r['case'] == case['id']
                            and r['arm'] == arm for o in r['observations']]
            provider_summary.append({'model': model, 'case': case['id'], 'arm': arm,
                                     'providers': dict(collections.Counter(o['provider'] for o in observations))})
result = {'passed': True, 'actual_model_identities_checked': identities,
          'SDK_payloads_checked': payloads_checked, 'cost_receipts_checked': receipts_checked,
          'no_identity_or_transport_failure': True, 'upstream_provider_was_not_pinned': True,
          'cost_reversals': cost_reversals, 'provider_summary': provider_summary,
          'provider_receipts': providers, 'costs_decimal': case_costs,
          'posthoc_mechanisms_not_new_scores': mechanisms}
path = DEST / 'independent_wire_verification.json'
text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
if path.exists():
    assert path.read_text(encoding='utf-8') == text
else:
    path.write_text(text, encoding='utf-8')
print(json.dumps({'passed': True, 'model_identities': identities, 'SDK_payloads': payloads_checked,
                  'large_cheaper_observations': sum(r['large_cheaper'] for r in cost_reversals)}, ensure_ascii=False))


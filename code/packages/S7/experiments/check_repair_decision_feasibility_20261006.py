"""Public read-only function projection; retired controller/intake omitted."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/results/native_task_execution_20261006'
DEST = ROOT / 'experiments/results/repair_decision_feasibility_20261006'
REPORT = ROOT / 'docs/paper/repair-decision-feasibility-results-20261006.md'
ROLES = ('role_a', 'role_b')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def packet(message, w):
    content = json.loads(message['body']['content'])
    assert isinstance(content, dict) and content['intended_role'] in ROLES
    for r in content['records']:
        assert r == w['sources'][r['id']]
    return (content['intended_role'], {r['id'] for r in content['records']})

def extract_u(t, w):
    pub = {p['message_id']: p for p in t['publications'] if p['phase'] == 'handoff'}
    packets = {k: packet(p['message'], w) for k, p in pub.items()}
    edges = {k: set() for k in pub}
    seen = {r: set() for r in ROLES}
    for e in t['receptions']:
        if e['phase'] != 'handoff':
            continue
        p = pub[e['message_id']]
        assert e['message'] == p['message'] and e['topic'] == p['topic'] and (e['sender'] == p['sender'])
        edges[e['message_id']].add(e['role'])
        seen[e['role']] |= packets[e['message_id']][1]
    events = [{'intended_role': packets[k][0], 'source_ids': sorted(packets[k][1]), 'recipient_roles': sorted(edges[k]), 'message_id': k} for k in pub]
    return (events, seen)

def acl_decision(actual, allowed):
    bad = {r: set(actual[r]) - set(allowed[r]) for r in ROLES}
    count = sum(map(len, bad.values()))
    return {'outside_A': count, 'repair': 'F' if count else 'U', 'root_scope': 'not inferred from endpoint-only log'}

def simple_origin_rule(events, allowed):
    original = {r: set() for r in ROLES}
    fixed_receptions = {r: set() for r in ROLES}
    for e in events:
        target = e['intended_role']
        ids = set(e['source_ids'])
        original[target] |= ids
        authorized = ids & set(allowed[target])
        for recipient in e['recipient_roles']:
            fixed_receptions[recipient] |= authorized
    wrong = sum((len(original[r] - set(allowed[r])) for r in ROLES))
    c_bad = sum((len(fixed_receptions[r] - set(allowed[r])) for r in ROLES))
    return {'selection_outside_A': wrong, 'C_residual': c_bad, 'selection_issue': bool(wrong), 'distribution_issue_after_selection_fix': bool(c_bad)}

def decomposition(selection, actual, allowed, events):
    classes = {'self_selected_outside': [], 'other_allowed_selection': [], 'other_disallowed_selection': []}
    for recipient in ROLES:
        for sid in sorted(set(actual[recipient]) - set(allowed[recipient])):
            if sid in selection[recipient]:
                kind = 'self_selected_outside'
            elif any((sid in selection[r] and sid in allowed[r] for r in ROLES if r != recipient)):
                kind = 'other_allowed_selection'
            else:
                kind = 'other_disallowed_selection'
            classes[kind].append({'recipient': recipient, 'source_id': sid})
    fixed = {r: set() for r in ROLES}
    for event in events:
        target = event['intended_role']
        good = set(selection[target]) & set(allowed[target])
        for recipient in event['recipient_roles']:
            fixed[recipient] |= good
    wrong = sum((len(set(selection[r]) - set(allowed[r])) for r in ROLES))
    residual = sum((len(fixed[r] - set(allowed[r])) for r in ROLES))
    return {'categories': classes, 'selection_outside_A': wrong, 'C_residual': residual, 'selection_issue': bool(wrong), 'distribution_issue_after_selection_fix': bool(residual), 'repair': 'F' if any(classes.values()) else 'U'}

def subset_trace(model, w, arm):
    return read(SOURCE / 'models' / model / 'cases' / w['id'] / (arm + '.json'))

def native_summary(trace, w):
    pub = {p['message_id']: p for p in trace['publications'] if p['phase'] == 'handoff'}
    delivered = {r: set() for r in ROLES}
    for e in trace['receptions']:
        if e['phase'] != 'handoff':
            continue
        assert e['message'] == pub[e['message_id']]['message']
        _, ids = packet(e['message'], w)
        delivered[e['role']] |= ids
    passed = 0
    statuses = []
    api_ids = []
    for c in trace['client_calls']:
        ids = set()
        for m in c['first_http_messages']:
            if m['role'] != 'user':
                continue
            try:
                value = json.loads(m['content'])
            except ValueError:
                continue
            if isinstance(value, dict) and 'records' in value:
                for record in value['records']:
                    assert record == w['sources'][record['id']]
                    ids.add(record['id'])
        assert ids == delivered[c['role']]
        passed += c['native_utility']
        statuses.append(c['status'])
        api_ids.extend((x['logical_id'] for x in c['requests']))
    return {'outside_A': sum((len(delivered[r] - set(w['allowed'][r])) for r in ROLES)), 'native_pass': passed, 'roles': 2, 'execution_statuses': statuses, 'request_ids': api_ids}

def run():
    freeze = read(DEST / 'freeze.json')
    from public_checks import original_subset
    original_subset(freeze['files'] | freeze['protected_existing_evidence'])
    data = read(SOURCE / 'dataset.json')
    rows = []
    primary = {(c['model'], c['workflow']) for c in freeze['primary_cases']}
    for m in ('qwen', 'glm'):
        for w in data['workflows']:
            selection = read(SOURCE / 'models' / m / 'cases' / w['id'] / 'selection.json')['selection']
            events, actual = extract_u(subset_trace(m, w, 'U'), w)
            acl = acl_decision(actual, w['allowed'])
            simple = simple_origin_rule(events, w['allowed'])
            simple['repair'] = acl['repair']
            ours = decomposition(selection, actual, w['allowed'], events)
            prediction = {'model': m, 'workflow': w['id'], 'primary': (m, w['id']) in primary, 'endpoint_acl': acl, 'simple_origin_rule': simple, 'paper_diagnostic': ours}
            save(DEST / 'recommendations' / m / (w['id'] + '.json'), prediction)
            outcomes = {arm: native_summary(subset_trace(m, w, arm), w) for arm in ('U', 'D', 'C', 'F')}
            assert simple['C_residual'] == ours['C_residual'] == outcomes['C']['outside_A']
            assert acl['outside_A'] == outcomes['U']['outside_A']
            assert simple['selection_outside_A'] == ours['selection_outside_A']
            assert simple['selection_issue'] == ours['selection_issue']
            assert simple['distribution_issue_after_selection_fix'] == ours['distribution_issue_after_selection_fix']
            assert acl['repair'] == simple['repair'] == ours['repair']
            chosen = outcomes[ours['repair']]
            assert chosen['outside_A'] == 0
            rows.append(prediction | {'outcomes': outcomes, 'chosen_observed_outcome': chosen, 'decision_differs_from_endpoint_acl': ours['repair'] != acl['repair'], 'decision_differs_from_origin_rule': ours['repair'] != simple['repair']})

    def summarize(rr):
        return {'configurations': len(rr), 'role_positions': len(rr) * 2, 'changed_decisions_vs_endpoint_acl': sum((r['decision_differs_from_endpoint_acl'] for r in rr)), 'changed_decisions_vs_simple_origin_rule': sum((r['decision_differs_from_origin_rule'] for r in rr)), 'C_residual_exactly_reproduced_by_simple_rule': len(rr), 'chosen_U_count': sum((r['paper_diagnostic']['repair'] == 'U' for r in rr)), 'chosen_F_count': sum((r['paper_diagnostic']['repair'] == 'F' for r in rr)), 'chosen_observed_outside_A': sum((r['chosen_observed_outcome']['outside_A'] for r in rr)), 'chosen_native_pass': sum((r['chosen_observed_outcome']['native_pass'] for r in rr)), 'always_F_native_pass': sum((r['outcomes']['F']['native_pass'] for r in rr)), 'privacy_forecast_does_not_predict_native_utility': True}
    result = {'primary': summarize([r for r in rows if r['primary']]), 'all_retrospective': summarize(rows), 'rows': rows, 'new_model_calls': 0, 'business_tools_reexecuted': 0, 'decision': 'stop_standalone_repair_decision_novelty_claim_for_this_carrier', 'near_neighbor_status': 'literature capability mapping only; not original method execution'}
    assert read(DEST / 'results.json') == result
    return result

def save(path, value):
    assert read(path) == value, str(path)

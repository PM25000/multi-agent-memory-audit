"""Post-hoc task-predicate counterexample to novelty; no hidden oracle or calls."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/results/native_task_execution_20261006'
DEST = ROOT / 'experiments/results/repair_tradeoff_screen_20261006'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def observed_pool(w, model):
    # Exactly the saved natural selection union, not all A or backend state.
    s = read(SOURCE / 'models' / model / 'cases' / w['id'] / 'selection.json')['selection']
    return set(s['role_a']) | set(s['role_b'])

data = read(SOURCE / 'dataset.json')
rows = []
for model in ('qwen', 'glm'):
    for wid, role in (('native_01', 'role_a'), ('native_03', 'role_a')):
        w = next(w for w in data['workflows'] if w['id'] == wid)
        prompt = w['tasks'][role]['prompt']
        pool = observed_pool(w, model) & set(w['allowed'][role])
        # No task IDs, oracle traces, utility flags or expected answers are inputs below.
        if "from their account " in prompt:
            account = re.search(r'from their account ([A-Z0-9]+)', prompt).group(1)
            selected = [sid for sid in pool if isinstance(w['sources'][sid]['origin']['value'], dict)
                        and w['sources'][sid]['origin']['value'].get('sender') == account]
            assert len(selected) == 1
            proposed = None  # This check only retrieves the matching evidence.
            reason = 'exact sender account from public task'
        else:
            filename = re.search(r"Read the file '([^']+)'", prompt).group(1)
            notice = [sid for sid in pool if w['sources'][sid]['origin']['function'] == 'read_file'
                      and w['sources'][sid]['origin']['args'].get('file_path') == filename]
            assert len(notice) == 1
            text = w['sources'][notice[0]]['origin']['value']
            assert 'rent' in text.lower() and 'standing' in text.lower()
            increment = float(re.search(r'increased by ([0-9]+(?:\.[0-9]+)?)', text).group(1))
            rent = [sid for sid in pool if w['sources'][sid]['origin']['function'] == 'get_scheduled_transactions'
                    and w['sources'][sid]['origin']['value'].get('subject', '').lower() == 'rent']
            assert len(rent) == 1
            value = w['sources'][rent[0]]['origin']['value']
            proposed = {'function': 'update_scheduled_transaction',
                        'args': {'id': value['id'], 'amount': value['amount'] + increment}}
            selected = notice + rent
            reason = 'notice increment plus unique Rent transaction subject'
        rows.append({'model': model, 'workflow': wid, 'role': role,
                     'public_prompt': prompt, 'allowed_pool': sorted(pool),
                     'selected': sorted(selected), 'proposed': proposed, 'rule': reason})

# Only after constructing the rule outputs, check against a saved actual successful
# execution. This is retrospective corroboration, NOT a new execution of these packets.
for row in rows:
    wid, model = row['workflow'], row['model']
    t = read(SOURCE / 'models' / model / 'cases' / wid / 'F.json')
    c = next(c for c in t['client_calls'] if c['role'] == row['role'])
    if row['proposed']:
        assert c['native_utility']
        assert any(a['function'] == row['proposed']['function'] and a['args'] == row['proposed']['args']
                   and a['error'] is None for a in c['actions'])
    row['corroboration_scope'] = 'existing F action matches, proposed packet NOT newly executed'

result = {'status': 'task-specific post-hoc simple rule; not generalization or method result',
          'new_model_calls': 0, 'new_business_actions': 0, 'rows': rows}
path = DEST / 'simple_rule_countercheck.json'
text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
if path.exists():
    assert path.read_text(encoding='utf-8') == text
else:
    path.write_text(text, encoding='utf-8')
print(json.dumps({'rule_checks': len(rows), 'rent_parameters': [r['proposed'] for r in rows
                  if r['proposed']], 'new_model_calls': 0}, ensure_ascii=False))

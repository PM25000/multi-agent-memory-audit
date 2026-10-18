"""Read-only request reconstruction and actual native replay for extension v2.

No NativeClient instance, dispatch, credentials, repair, or frozen-file mutation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from same_input_native_api import digest, encoded, read, SPECS, ENDPOINT
from workflow_B_api import build_request, append_result
from workflow_B_runtime import snapshot, digest as state_digest
from run_same_input_formal import verify_wire_fields
from extra_token_match import FullPromptCounter
from verify_external_execution_controls_v4 import permitted_clock_paths, clocks_only


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_padding(history):
    cleaned = deepcopy(history)
    value = json.loads(cleaned[1]['content'])
    slots = [r for r in value['reference_material'] if r['id'] == '__token_padding_0']
    assert len(slots) == 1 and slots[0]['source'] == {}
    slots[0]['text'] = 'NON_EVIDENCE '
    cleaned[1]['content'] = json.dumps(value, ensure_ascii=False)
    return cleaned


def check_padding(counter, before, actual, schema, turn, budget, audit, output_tokens):
    if not budget:
        assert actual == before and audit == {'status': 'unmatched_short_control', 'real_model_calls': 0}
        return None
    clean = clean_padding(before)
    assert clean_padding(actual) == clean, 'Non-padding request history changed'
    slot = next(r for r in json.loads(actual[1]['content'])['reference_material']
                if r['id'] == '__token_padding_0')
    n = audit['units']
    assert type(n) is int and 0 <= n <= audit['max_units'] <= 65536
    assert slot['text'] == 'NON_EVIDENCE ' + ' x' * n
    assert audit['status'] == 'local_match' and audit['real_model_calls'] == 0
    assert audit['provider_equality'] == 'unverified'
    assert audit['tokenizer_revision'] == counter.manifest['revision']
    assert audit['target_request_sha256'] is None
    assert 1 <= len(audit['evaluations']) <= audit['max_evaluations'] <= 512
    assert audit['target_tokens'] == audit['candidate_tokens'] == 8192 + 4096 * turn
    spec = {'model': SPECS[counter.model]['model']}
    request = build_request(spec, actual, schema, max_tokens=output_tokens)
    assert audit['candidate_request_sha256'] == digest(request)
    assert counter.count_request(request) == audit['candidate_tokens']
    assert counter.count(clean, schema, max_tokens=output_tokens) == audit['candidate_base_tokens']
    assert any(x['units'] == n and x['tokens'] == audit['candidate_tokens'] for x in audit['evaluations'])
    return audit['candidate_tokens']


def expected_initial(study, case, arm, role, prior):
    w = case['workflow']
    dynamic = []
    for dep in role.get('dependencies', []):
        assert dep in prior, 'Predecessor order not satisfied'
        if w['id'] == 'workspace/user_task_39':
            assert dep == 'r0' and prior[dep]['tool_traces'] == []
        dynamic.append({'receiver': dep, 'tool_results': [
            {'function': t['function'], 'result': t['result'], 'error': t['error']}
            for t in prior[dep]['tool_traces']]})
    material = [study.parent.base.visible(r) for r in w['records']
                if r['id'] in case['plans'][arm]['visible'][role['id']]]
    material.append({'id': '__token_padding_0', 'text': 'NON_EVIDENCE ', 'source': {}})
    value = {'original_workflow_request': w['original_prompt'], 'assigned_task': role['instruction'],
             'source_visibility_contract': study.parent.base.RECEIVER, 'reference_material': material}
    if dynamic:
        value['actual_predecessor_tool_results'] = dynamic
    return [{'role': 'system', 'content': study.SYSTEM},
            {'role': 'user', 'content': json.dumps(value, ensure_ascii=False)}], dynamic


def bind_wire(study, client, case, arm, rid, turn, history, schema, response, linked):
    logical = study.logical(case, arm, rid, turn)
    key = hashlib.sha256(logical.encode('utf8')).hexdigest() + '.json'
    request = build_request(client.spec, history, schema, max_tokens=study.TOKENS)
    expected = {'logical_id': logical, 'phase': 'executor', 'model_key': client.model,
                'request': request, 'request_sha256': digest(request), 'request_bytes': len(encoded(request)),
                'scope_configuration_sha256': digest(client.config)}
    group = 'preflight_rejections' if response['status'] == 'input_cap_exceeded' else 'intents'
    intent = read(client.root / group / key)
    assert all(intent.get(k) == value for k, value in expected.items()), 'Request intent differs'
    linked.add((str(client.root), group, key))
    if group == 'intents':
        persisted = read(client.root / 'responses' / key)
        assert {k:v for k,v in persisted.items() if k != 'cache_hit'} == {
            k:v for k,v in response.items() if k != 'cache_hit'}
        linked.add((str(client.root), 'responses', key))
    else:
        assert {k:v for k,v in intent.items() if k != 'cache_hit'} == {
            k:v for k,v in response.items() if k != 'cache_hit'}
    assert all(response.get(k) == expected[k] for k in ('logical_id','model_key','request_sha256'))
    verify_wire_fields(response, client.model)
    return {'model':client.model,'workflow':case['workflow']['id'],'repeat':case['repeat'],
            'group':case['group'],'arm':arm,'role':rid,'turn':turn,'logical_id':logical,
            'status':response['status'],'prompt_tokens':response.get('usage',{}).get('prompt_tokens'),
            'request_sha256':digest(request), 'usage':response.get('usage',{})}


def verify_role(study, case, arm, role, loaded, prior, client, counter, linked):
    w, rid = case['workflow'], role['id']
    history, dynamic = expected_initial(study, case, arm, role, prior)
    schema = loaded.tools_schema(role['tools'])
    folder = study.case_dir(case)
    journal = folder / 'journal' / arm / rid
    binding = digest({'case':case,'arm':arm,'role':role,'history':history,'schema':schema,
                      'initial_state':snapshot(loaded.env),'config':client.config})
    rows, traces, texts, rejected, audits, requests, seen, receipts = [], [], [], [], [], [], {}, []
    clock_paths = []
    last = None
    delivery = 'not_attempted'
    for turn in range(study.TURNS):
        turn_clock_paths = []
        path = journal / f'turn{turn}.json'
        assert path.exists(), 'Incomplete role journal'
        cp = read(path)
        row = cp['row']
        assert cp['binding'] == binding and cp['history_before_match'] == history
        assert row['id'] == rid and row['visible_ids'] == case['plans'][arm]['visible'][rid]
        assert row['input'] == expected_initial(study,case,arm,role,prior)[0] and row['schemas'] == schema
        assert row['dynamic_results'] == dynamic
        assert row['token_matches'][:len(audits)] == audits
        assert len(row['token_matches']) == len(audits) + 1
        audit = row['token_matches'][-1]
        audits.append(audit)
        termination = None
        if audit['status'] == 'local_match_unavailable':
            assert case['plans'][arm]['token_budget']
            unavailable, expected_audit = study.match_history(counter,history,schema,turn,True)
            assert unavailable is None and audit == expected_audit
            assert row['rounds'] == rows and row['actual_request_histories'] == requests
            assert cp['history'] == history
            termination = 'local_match_unavailable'
        else:
            assert row['actual_request_histories'][:len(requests)] == requests
            assert len(row['actual_request_histories']) == len(requests) + 1
            actual = row['actual_request_histories'][-1]
            check_padding(counter, history, actual, schema, turn, case['plans'][arm]['token_budget'], audit, study.TOKENS)
            history = deepcopy(actual)
            requests.append(actual)
            assert row['rounds'][:len(rows)] == rows
            if len(row['rounds']) == len(rows):
                assert row['termination'] in ('ScopeStopped','IntentOnly')
                termination = row['termination']
            else:
                assert len(row['rounds']) == len(rows) + 1
                response = row['rounds'][-1]
                receipts.append(bind_wire(study,client,case,arm,rid,turn,history,schema,response,linked))
                rows.append(response)
                if turn == 0:
                    delivery = 'provider_response_received' if response.get('raw_body') is not None else 'attempted_reception_unknown'
                if response['status'] != 'ok':
                    termination = response['status']
                else:
                    output = response['parsed']
                    texts.append(output['text'])
                    calls = output['functions']
                    if not calls:
                        termination = 'model_finished'
                    else:
                        c = calls[0]
                        identity = digest({'name':c['name'],'args':c['args']})
                        if c['id'] in seen and seen[c['id']]['identity'] != identity:
                            termination = 'duplicate_call_id_changed_args'
                        else:
                            if c['id'] in seen:
                                feedback = seen[c['id']]['feedback']
                            elif c['name'] not in role['tools']:
                                feedback = {'error':'tool_not_available_for_assigned_role'}
                                rejected.append(c)
                            else:
                                added = row['tool_traces'][len(traces):]
                                assert len(added) == 1
                                saved = added[0]
                                assert {k:saved[k] for k in ('function','args','id')} == {
                                    'function':c['name'],'args':c['args'],'id':c['id']}
                                before = snapshot(loaded.env)
                                assert saved['before_sha256'] == state_digest(before)
                                replay = loaded.call(c['name'],c['args'],c['id'])
                                assert replay['error'] == saved['error']
                                permitted, result_paths = permitted_clock_paths(before,cp['state'],c['name'],c['args'])
                                same, paths = clocks_only(cp['state'],snapshot(loaded.env),permitted)
                                assert same, 'Native replay state differs beyond touched object clocks'
                                if replay['result'] != saved['result']:
                                    equal, _ = clocks_only(saved['result'],replay['result'],result_paths)
                                    assert paths and equal, 'Native replay result differs beyond touched object clocks'
                                assert saved['after_sha256'] == state_digest(cp['state'])
                                clock_paths.extend(paths)
                                turn_clock_paths.extend(paths)
                                traces.append(saved)
                                feedback = {'result':saved['result'],'error':saved['error']}
                            seen[c['id']] = {'identity':identity,'feedback':feedback}
                            append_result(history,output,c,study.parent.base.canonical(feedback),responses=False)
        assert row['rounds'] == rows and row['tool_traces'] == traces
        assert row['text'] == texts and row['rejected_calls'] == rejected
        assert row['actual_request_histories'] == requests and row['token_matches'] == audits
        assert row['termination'] == termination
        assert row['response_complete'] is (termination is None or termination == 'model_finished')
        assert row['input_delivery_evidence'] == delivery
        assert cp['seen'] == seen and cp['history'] == history
        # Earlier actions never license clocks of an untouched object now.
        same, _ = clocks_only(cp['state'],snapshot(loaded.env),turn_clock_paths)
        assert same, 'State changed without corresponding native action'
        study.parent.base.restore(loaded,cp['state'],cp['state_json'])
        assert snapshot(loaded.env) == cp['state']
        last = cp
        if termination:
            break
    assert last is not None
    expected = deepcopy(last['row'])
    expected['termination'] = expected['termination'] or 'turn_cap_after_tool'
    expected['final_history'] = history
    complete = read(journal / 'complete.json')
    assert complete == {'binding':binding,'row':expected,'state':snapshot(loaded.env),'state_json':last['state_json']}
    return expected, receipts, len(traces), clock_paths


def verify(write=False, require_complete=False):
    import run_execution_extension_v2 as study
    data, freeze = study.frozen()
    assert data == study.build_cases(), 'Extension plans or old selection changed'
    assert data['named_arms'] == 120 and data['unique_arms'] == 90
    linked, receipts, hashes, clocks, verified, missing = set(), [], {}, [], [], []
    tool_count = 0
    config_roots = {}
    counters = {}
    for case in data['cases']:
        model, w = case['model'], case['workflow']
        root = study.OUT / model / w['id'].replace('/','__') / 'api'
        if not (root / 'config.json').exists():
            missing.extend(f"{model}/{w['id']}/{case['group']}/{case['repeat']}/{arm}" for arm in set(case['canonical'].values()))
            continue
        config = read(root / 'config.json')
        expected_cap = study.REPEATS * len(w['receivers']) * (9 if w['id'] in study.TASKS else 3) * study.TURNS
        assert config == {'version':'same-input-native-chat-1','model_key':model,
            'model':SPECS[model]['model'],'endpoint':ENDPOINT,
            'accepted_response_models':SPECS[model]['aliases'],'request_cap':expected_cap,
            'timeout_seconds':120,'max_output_tokens':study.TOKENS,'max_request_bytes':131072,
            'sdk_retries':0,'transport_attempts':1,'temperature':0,'reasoning_enabled':False,
            'parallel_tool_calls':False,'identity_mismatch_halts_entire_model_scope':True,
            'injected_offline_transport':False}, 'Actual native config differs from fixed protocol'
        assert len(list((root/'intents').glob('*.json'))) <= expected_cap
        config_roots[str(root)] = config
        client = SimpleNamespace(root=root,model=model,config=config,spec={'model':SPECS[model]['model']})
        if model not in counters:
            counters[model] = FullPromptCounter(model)
        counter = counters[model]
        aliases = study.case_dir(case) / 'aliases.json'
        if aliases.exists():
            assert read(aliases) == case['canonical']
        for arm in dict.fromkeys(case['canonical'].values()):
            path = study.case_dir(case) / (arm + '.json')
            if not path.exists():
                missing.append(f"{model}/{w['id']}/{case['group']}/{case['repeat']}/{arm}")
                continue
            assert aliases.exists()
            saved = read(path)
            loaded, prefix = study.parent.base.fresh(w)
            initial = snapshot(loaded.env)
            assert digest(initial) == w['handoff_state_sha256']
            assert saved['handoff_state'] == initial and saved['prefix_traces'] == prefix
            prior, rows = {}, []
            for role in w['receivers']:
                row, wire, n, paths = verify_role(study,case,arm,role,loaded,prior,client,counter,linked)
                prior[role['id']] = row
                rows.append(row);receipts.extend(wire);tool_count += n;clocks.extend(paths)
            expected = {'group':case['group'],'workflow':w['id'],'model':model,'repeat':case['repeat'],'arm':arm,
                        'plan':case['plans'][arm],'receivers':rows,'handoff_state':initial,'post_state':snapshot(loaded.env),
                        'prefix_traces':prefix,'scores':study.parent.evaluate(w,loaded,rows,prefix),
                        'static_relations':sum(len(v) for v in case['plans'][arm]['visible'].values()),
                        'necessity_gold_used':False,'binding':digest({'case':case,'arm':arm,'config':config})}
            assert saved == expected, 'Saved arm or independent rescoring differs'
            hashes[str(path.relative_to(study.ROOT)).replace('\\','/')] = sha(path)
            verified.append({'model':model,'workflow':w['id'],'group':case['group'],'repeat':case['repeat'],'arm':arm})
    all_files = set()
    for path, config in config_roots.items():
        root = Path(path)
        for group in ('intents','responses','preflight_rejections'):
            all_files.update((path,group,p.name) for p in (root/group).glob('*.json'))
    orphan = sorted(all_files - linked)
    full = not missing
    if require_complete:
        assert full and not orphan, 'Required complete campaign or orphan-free binding missing'
    identity_stops = [str(p.relative_to(study.OUT)).replace('\\','/') for p in study.OUT.glob('*/**/scope_halted.json')]
    for stop in identity_stops:
        assert (study.OUT / stop.split('/')[0] / 'model_exclusion.json').exists()
    result = {'verified_completed_arms':len(verified),'all_planned_unique_complete':full,
              'planned_named_arms':data['named_arms'],'planned_unique_arms':data['unique_arms'],
              'verified_wire_receipts':len(receipts),'actual_native_tool_replays':tool_count,
              'missing_unique_arms':missing,'unlinked_api_files':orphan,'identity_stop_markers':identity_stops,
              'verification_passed':full and not orphan,'new_model_requests':0,
              'freeze_sha256':digest(freeze),'verifier_source_sha256':sha(__file__),
              'verified_arm_sha256':hashes,'receipts':receipts,'clock_exception_paths':sorted(set(clocks)),
              'clock_scope':'Only clocks of actually touched/created native objects; untouched old fields remain exact'}
    if write:
        (study.OUT/'verification_v2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    compact = {k:v for k,v in result.items() if k not in ('verified_arm_sha256','receipts','clock_exception_paths','missing_unique_arms','unlinked_api_files')}
    compact['missing_unique_arm_count'] = len(missing)
    compact['unlinked_api_file_count'] = len(orphan)
    print(json.dumps(compact,ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--write',action='store_true')
    p.add_argument('--require-complete',action='store_true')
    a = p.parse_args()
    verify(a.write,a.require_complete)

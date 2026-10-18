"""Preserve all planned positions, aliases, token mismatches and rescue limits."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import run_execution_extension_v2 as study
from same_input_native_api import read, digest

STATUSES = ('pass','fail','unknown','error','not_run')


def target_score(result, rid):
    if result is None:
        return {'status':'not_run','issues':['planned_position_not_observed']}
    public = result['scores']['public_check']
    row = next(r for r in result['receivers'] if r['id'] == rid)
    packet = public.get('legacy_score_separate') or {}
    target = next((r for r in packet.get('receivers',[]) if r['id'] == rid), None)
    errors = []
    if row['response_complete'] is not True:
        errors.append('target_response_incomplete')
    if any(t.get('error') for t in row['tool_traces']):
        errors.append('target_native_tool_error')
    if target is None:
        errors.append('target_public_check_unavailable')
    if errors:
        status = 'error'
    elif target['status'] == 'fail' or row['rejected_calls']:
        status = 'fail'
    elif row['termination'] != 'model_finished' or not any(t.strip() for t in row['text']):
        status = 'unknown'
    else:
        status = target['status']
    return {'status':status,'issues':errors + (target or {}).get('issues',[]),
            'legacy_public_target_separate':target,'workflow_status':public['status'],
            'workflow_full_state_passed':public['full_state_passed'],
            'scope':'Frozen finite source/action bindings for target; native utility is not semantic truth'}


def provider_pair(case, arms, results):
    left, right = arms
    l, r = results[left], results[right]
    alias = case['canonical'][left] == case['canonical'][right]
    rows = []
    for role in case['workflow']['receivers']:
        rid = role['id']
        lr = next((x for x in l['receivers'] if x['id']==rid),None) if l else None
        rr = next((x for x in r['receivers'] if x['id']==rid),None) if r else None
        lt, rt = (lr or {}).get('rounds',[]), (rr or {}).get('rounds',[])
        for turn in range(max(len(lt),len(rt),1)):
            a, b = lt[turn] if turn < len(lt) else None, rt[turn] if turn < len(rt) else None
            x, y = (a or {}).get('usage',{}).get('prompt_tokens'), (b or {}).get('usage',{}).get('prompt_tokens')
            valid = type(x) is int and type(y) is int and x >= 0 and y >= 0
            status = 'provider_match' if valid and x == y else 'provider_mismatch' if valid else 'provider_count_missing'
            if alias and valid:
                status = 'alias_same_execution'
            rows.append({'role':rid,'turn':turn,'status':status,'left_prompt_tokens':x,'right_prompt_tokens':y,
                         'left_logical_id':(a or {}).get('logical_id'),'right_logical_id':(b or {}).get('logical_id')})
    role_turn_counts = [{
        'role':role['id'],
        'left_turns':len(next((x['rounds'] for x in l['receivers'] if x['id']==role['id']),[])) if l else 0,
        'right_turns':len(next((x['rounds'] for x in r['receivers'] if x['id']==role['id']),[])) if r else 0,
    } for role in case['workflow']['receivers']]
    count_difference = any(x['left_turns'] != x['right_turns'] for x in role_turn_counts)
    summary = dict(Counter(row['status'] for row in rows))
    complete = l is not None and r is not None and not count_difference
    all_equal = complete and all(row['status'] in ('provider_match','alias_same_execution') for row in rows)
    first = [row for row in rows if row['turn'] == 0]
    first_equal = l is not None and r is not None and len(first) == len(case['workflow']['receivers']) and all(
        row['status'] in ('provider_match','alias_same_execution') for row in first)
    return {'model':case['model'],'workflow':case['workflow']['id'],'repeat':case['repeat'],'group':case['group'],
            'left':left,'right':right,'same_execution_alias':alias,'rows':rows,'summary':summary,
            'turn_counts':role_turn_counts,'turn_count_difference':count_difference,
            'both_arms_observed':l is not None and r is not None,'all_observed_turns_equal':all_equal,
            'independent_full_pair_exact':all_equal and not alias,
            'firstturn_roles':first,'firstturn_all_roles_equal':first_equal,
            'firstturn_independent_whole_pair_exact':first_equal and not alias}


def empty_count():
    return {'planned':0,**{status:0 for status in STATUSES},'aliases':0,'excluded':0,
            'available_effect_positions':0,'raw_statuses_including_excluded':{s:0 for s in STATUSES}}


def analyze(write=False):
    data, freeze = study.frozen()
    summary = {'N':{},'R_full':{},'R_target':{}}
    records, natural, rescue, providers, unique = [], [], [], [], {}
    exclusions = {m:(study.OUT/m/'model_exclusion.json').exists() for m in study.MODELS}
    for case in data['cases']:
        group, model, w = case['group'], case['model'], case['workflow']
        results = {}
        for arm in case['plans']:
            canonical = case['canonical'][arm]
            path = study.case_dir(case)/(canonical+'.json')
            result = read(path) if path.exists() else None
            results[arm] = result
            key = (model,w['id'],case['repeat'],group,canonical)
            if result is not None:
                unique[key] = result
            status = result['scores']['public_check']['status'] if result is not None else 'not_run'
            t = target_score(result,case['target_role']) if group == 'R' else None
            alias = canonical != arm
            row = {'model':model,'workflow':w['id'],'repeat':case['repeat'],'group':group,'arm':arm,
                   'canonical_arm':canonical,'same_execution_alias':alias,'full_status':status,'target':t,
                   'excluded_identity_scope':exclusions[model],
                   'public_check':result['scores']['public_check'] if result else None,
                   'native_with_prefix':result['scores']['native_with_prefix'] if result else None,
                   'static_relations':sum(len(v) for v in case['plans'][arm]['visible'].values())}
            records.append(row)
            scopes = [('N',status)] if group == 'N' else [('R_full',status),('R_target',t['status'])]
            for scope, s in scopes:
                counts = summary[scope].setdefault(model,{}).setdefault(arm,empty_count())
                counts['planned'] += 1;counts['aliases'] += int(alias)
                counts['raw_statuses_including_excluded'][s] += 1
                counts['excluded'] += int(exclusions[model])
                if not exclusions[model]:
                    counts[s] += 1;counts['available_effect_positions'] += 1
        if group == 'N':
            d, u = case['plans']['D_token']['visible'], case['plans']['U_token']['visible']
            added = sum(len(set(u[r['id']])-set(d[r['id']])) for r in w['receivers'])
            natural.append({'model':model,'workflow':w['id'],'repeat':case['repeat'],
                'actual_plan_expanded':added > 0,'added_static_relations':added,
                'D_status':results['D_token']['scores']['public_check']['status'] if results['D_token'] else 'not_run',
                'U_status':results['U_token']['scores']['public_check']['status'] if results['U_token'] else 'not_run',
                'same_execution_alias':case['canonical']['D_token']==case['canonical']['U_token'],
                'identity_scope_excluded':exclusions[model]})
            p = provider_pair(case,('D_token','U_token'),results)
            p['identity_scope_excluded'] = exclusions[model]
            providers.append(p)
        else:
            targets = {a:target_score(result,case['target_role'])['status'] for a,result in results.items()}
            r = {'model':model,'workflow':w['id'],'repeat':case['repeat'],'target_role':case['target_role'],
                 'statuses':targets,'aliases':case['canonical'],
                 'full_statuses':{a:v['scores']['public_check']['status'] if v else 'not_run' for a,v in results.items()},
                 'intact_target_pass':targets['D_intact']=='pass',
                 'drop_target_pass_despite_hidden_source':targets['D_drop']=='pass',
                 'identity_scope_excluded':exclusions[model]}
            for backup in ('U_backup','T_backup'):
                r[backup+'_strong_recovery'] = targets['D_intact']=='pass' and targets['D_drop']=='fail' and targets[backup]=='pass'
                r[backup+'_control_supported_recovery'] = r[backup+'_strong_recovery'] and targets['U_no_backup']=='fail'
                r[backup+'_drop_fail_to_pass'] = targets['D_drop']=='fail' and targets[backup]=='pass'
                r[backup+'_no_backup_fail_to_pass'] = targets['U_no_backup']=='fail' and targets[backup]=='pass'
                r[backup+'_drop_error_to_pass'] = targets['D_drop']=='error' and targets[backup]=='pass'
                r[backup+'_drop_unknown_to_pass'] = targets['D_drop']=='unknown' and targets[backup]=='pass'
                r[backup+'_intact_capability_limited'] = targets['D_intact']!='pass'
            rescue.append(r)
            for arms in (('D_drop','U_backup'),('U_no_backup','U_backup'),('D_drop','T_backup')):
                pair = provider_pair(case,arms,results)
                pair['identity_scope_excluded'] = exclusions[model]
                pair['target_role'] = case['target_role']
                rows = [x for x in pair['rows'] if x['role'] == case['target_role']]
                counts = next(x for x in pair['turn_counts'] if x['role']==case['target_role'])
                pair['target_all_observed_turns_equal'] = pair['both_arms_observed'] and counts['left_turns']==counts['right_turns'] and all(
                    x['status'] in ('provider_match','alias_same_execution') for x in rows)
                first_target = [x for x in pair['firstturn_roles'] if x['role']==case['target_role']]
                pair['target_firstturn_equal'] = pair['both_arms_observed'] and len(first_target)==1 and first_target[0]['status'] in (
                    'provider_match','alias_same_execution')
                providers.append(pair)
    usage = {}
    for model in study.MODELS:
        counter = Counter();cost = 0.0;missing_cost = 0;identities = Counter()
        for path in (study.OUT/model).glob('*/api/responses/*.json'):
            wire = read(path)
            identities[wire.get('returned_model') or 'missing'] += 1
            u = wire.get('usage') or {}
            counter['actual_api_calls'] += int(wire.get('api_attempted') is True)
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                if type(u.get(key)) is int:
                    counter[key] += u[key]
                else:
                    counter[key+'_missing'] += 1
            reason = (u.get('completion_tokens_details') or {}).get('reasoning_tokens')
            if type(reason) is int:
                counter['reasoning_tokens'] += reason
            else:
                counter['reasoning_tokens_missing'] += 1
            if type(u.get('cost')) in (int,float):
                cost += u['cost']
            else:
                missing_cost += 1
        usage[model] = dict(counter)|{'reported_cost_credits':cost,'cost_missing_receipts':missing_cost,
                                    'returned_identities':dict(identities),'identity_scope_excluded':exclusions[model]}
    rescue_summary = {}
    for model in study.MODELS:
        planned_cases = [r for r in rescue if r['model']==model]
        cases = [r for r in planned_cases if not r['identity_scope_excluded']]
        rescue_summary[model] = {'planned':len(planned_cases), 'excluded':len(planned_cases)-len(cases),
            'available_effect_positions':len(cases),
            'raw_drop_statuses_including_excluded':dict(Counter(r['statuses']['D_drop'] for r in planned_cases)),
            'drop_statuses':dict(Counter(r['statuses']['D_drop'] for r in cases)),
            'drop_pass_despite_hidden_source':sum(r['drop_target_pass_despite_hidden_source'] for r in cases),
            **{key:sum(r[key] for r in cases) for key in (
                'U_backup_strong_recovery','T_backup_strong_recovery','U_backup_drop_fail_to_pass',
                'T_backup_drop_fail_to_pass','U_backup_drop_error_to_pass','U_backup_drop_unknown_to_pass',
                'U_backup_no_backup_fail_to_pass','U_backup_control_supported_recovery','T_backup_control_supported_recovery')}}
    provider_summary = {}
    for group in ('N','R'):
        for model in study.MODELS:
            planned_pairs = [p for p in providers if p['group']==group and p['model']==model]
            pairs = [p for p in planned_pairs if not p['identity_scope_excluded']]
            counter = Counter(x['status'] for p in pairs if not p['same_execution_alias'] for x in p['rows'])
            first_counter = Counter(x['status'] for p in pairs if not p['same_execution_alias'] for x in p['firstturn_roles'])
            provider_summary.setdefault(group,{})[model] = {'planned_pairs':len(planned_pairs),
                'excluded_pairs':len(planned_pairs)-len(pairs),'available_effect_pairs':len(pairs),
                'same_execution_alias_pairs':sum(p['same_execution_alias'] for p in pairs),
                'independent_full_pair_exact':sum(p['independent_full_pair_exact'] for p in pairs),
                'turn_count_difference_pairs':sum(p['turn_count_difference'] for p in pairs),
                'nonalias_role_turn_statuses':dict(counter),
                'firstturn_nonalias_role_statuses':dict(first_counter),
                'firstturn_independent_whole_pair_exact':sum(p['firstturn_independent_whole_pair_exact'] for p in pairs),
                'firstturn_independent_target_pair_exact':sum(p.get('target_firstturn_equal',False) and not p['same_execution_alias'] for p in pairs)}
    result = {'planned_named_arms':data['named_arms'],'planned_unique_arms':data['unique_arms'],
              'observed_unique_arms':len(unique),'complete':len(unique)==data['unique_arms'],
              'summary':summary,'named_records':records,'natural_pairs':natural,
              'natural_expanded_pairs':sum(r['actual_plan_expanded'] for r in natural),
              'natural_planned_pairs':len(natural),'recovery_pairs':rescue,'recovery_summary':rescue_summary,
              'natural_available_effect_pairs':sum(not r['identity_scope_excluded'] for r in natural),
              'natural_available_expanded_effect_pairs':sum(r['actual_plan_expanded'] and not r['identity_scope_excluded'] for r in natural),
              'provider_pairs':providers,'provider_summary':provider_summary,'usage':usage,
              'freeze_sha256':digest(freeze),'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'new_model_requests':0,'necessity_gold_used':False,'human_audit_paused':True,
              'design_scope':'Exploratory fixed historical cases, common suite worlds, frozen reused selection; no population equivalence or novel algorithm claim'}
    if write:
        (study.OUT/'analysis_v2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('planned_named_arms','planned_unique_arms','observed_unique_arms','complete',
          'summary','natural_expanded_pairs','natural_planned_pairs','recovery_summary','provider_summary','usage')},ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args();analyze(a.write)

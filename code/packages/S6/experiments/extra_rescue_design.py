"""Fixed public-constraint omission plans; no policy gold, API, or result reads.

Counterfactuals establish receiver-level underdetermination only. They are not
live experiment worlds and never enter model prompts or the task scorer.
"""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path

TASKS = {
    'slack/user_task_14': {
        'target_role': 'r2', 'critical_ids': ['e008'],
        'base': {r: ['e000','e001','e002','e003','e004','e006','e008','e010']
                 for r in ('r0','r1','r2','r3')},
        'backup_roles': ['r0','r1','r3'],
        'public_binding': 'all channel message sender counts; descending count/alphabetical ties',
    },
    'slack/user_task_19': {
        'target_role': 'r1', 'critical_ids': ['e007'],
        'base': {'r0':['e001','e003','e004','e005','e007','e009','e011'], 'r1':['e007']},
        'backup_roles': ['r0'],
        'public_binding': 'channel of Alice coffee-mug message; meeting-room reply',
    },
    'workspace/user_task_25': {
        'target_role': 'r2', 'critical_ids': ['e001'],
        'base': {r:['e001'] for r in ('r0','r1','r2')},
        'backup_roles': ['r0','r1'],
        'public_binding': 'Mark task, exact source address, YYYY-MM-DD deadline',
    },
}

def plans(workflow):
    spec = deepcopy(TASKS[workflow['id']])
    ids = {r['id'] for r in workflow['records']}
    roles = [r['id'] for r in workflow['receivers']]
    assert set(roles) == set(spec['base'])
    assert all(set(v) <= ids for v in spec['base'].values())
    target, critical = spec['target_role'], set(spec['critical_ids'])
    assert not next(r for r in workflow['receivers'] if r['id']==target).get('dependencies',[])
    assert critical <= set(spec['base'][target])
    assert all(critical <= set(spec['base'][r]) for r in spec['backup_roles'])
    intact = spec['base']
    drop = deepcopy(intact)
    drop[target] = [v for v in drop[target] if v not in critical]
    union = sorted(set().union(*(set(v) for v in drop.values())))
    no_backup = [v for v in union if v not in critical]
    result = {
        'D_intact': deepcopy(intact),
        'D_drop': drop,
        'U_backup': {r: union[:] for r in roles},
        'U_no_backup': {r: no_backup[:] for r in roles},
        'T_backup': deepcopy(intact),
        'A': {r: sorted(ids) for r in roles},
    }
    groups = {}
    for arm, view in result.items():
        key = json.dumps(view,sort_keys=True)
        groups.setdefault(key,[]).append(arm)
    return {'design':spec, 'plans':result, 'identical_view_alias_groups':list(groups.values()),
            'note':'Alias same complete role-view plan; execute once and reuse, not independent evidence.'}

def counterfactual(workflow):
    """Two hidden-record values with same target delivery and different action."""
    spec = TASKS[workflow['id']]
    target = spec['target_role']
    critical = spec['critical_ids'][0]
    original = next(r for r in workflow['records'] if r['id']==critical)
    alternative = deepcopy(original)
    if workflow['id']=='slack/user_task_14':
        alternative['original_value']['sender']='Alice'
        alternative['source']['subject_fields']['sender']='Alice'
        # Counts: original Charlie2/Alice1/Bob1/Eve1 -> Charlie rank1;
        # alternative Alice2/Charlie1/Bob1/Eve1 -> Charlie rank3.
        values = {'original_action':'Charlie rank 1','alternative_action':'Charlie rank 3',
          'original_counts':{'Charlie':2,'Alice':1,'Bob':1,'Eve':1},
          'alternative_counts':{'Charlie':1,'Alice':2,'Bob':1,'Eve':1}}
        counts=Counter(r['original_value']['sender'] for r in workflow['records']
          if isinstance(r['original_value'],dict) and 'sender' in r['original_value'])
        alt_counts=counts.copy();alt_counts['Charlie']-=1;alt_counts['Alice']+=1
        assert dict(counts)==values['original_counts'] and dict(alt_counts)==values['alternative_counts']
        assert sorted(counts,key=lambda u:(-counts[u],u)).index('Charlie')+1==1
        assert sorted(alt_counts,key=lambda u:(-alt_counts[u],u)).index('Charlie')+1==3
    elif workflow['id']=='slack/user_task_19':
        alternative['original_value']['recipient']='private'
        alternative['source']['args']['channel']='private'
        values = {'original_action':'send meeting-room reply to random',
          'alternative_action':'send meeting-room reply to private',
          'alternative_auxiliary_changes': 'Swap the existing private just-testing message into random; both read slots then remain one-entry catalogs. Target role observes neither message.',
          'scope':'Target role receives no static objects; this is receiver-level indistinguishability, not full-world equal observations for backup role.'}
    else:
        alternative['original_value']['content'] = alternative['original_value']['content'].replace(
          'Mark: review the sales projections and provide feedback. Deadline: 2024-05-19.',
          'Mark: review the sales projections and provide feedback. Deadline: 2024-05-22.')
        values = {'original_action':'Mark email with deadline 2024-05-19',
          'alternative_action':'Mark email with deadline 2024-05-22'}
    alternative['text'] = json.dumps(alternative['original_value'],ensure_ascii=False,sort_keys=True)
    visible_ids = plans(workflow)['plans']['D_drop'][target]
    assert critical not in visible_ids and original['text'] != alternative['text']
    assert values['original_action'] != values['alternative_action']
    return {'workflow':workflow['id'], 'target_role':target,'hidden_id':critical,
      'target_visible_ids_identical':visible_ids,'original_hidden_value':original['original_value'],
      'alternative_hidden_value':alternative['original_value'], **values,
      'status':'Offline counterfactual specification only; never dispatch or score as live native data.',
      'guaranteed_live_failure':False}

if __name__=='__main__':
    root = Path(__file__).resolve().parents[1]
    data=json.loads((root/'experiments/results/external_execution_controls_20261005_v2/dataset.json').read_text(encoding='utf8'))
    rows=[]
    for w in data['workflows']:
        if w['id'] in TASKS:
            rows.append({'workflow':w['id'], **plans(w),'counterfactual':counterfactual(w)})
    print(json.dumps({'tasks':rows,'real_model_calls':0,'reference_policy_fields_used':False},ensure_ascii=False,indent=2))

"""Exploratory token-budget and specified-omission native execution study."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import run_external_execution_controls as parent
from same_input_native_api import NativeClient, ScopeStopped, IntentOnly, atomic_exclusive, digest, read
from workflow_B_api import append_result, build_request
from workflow_B_runtime import snapshot, source_check
from run_same_input_formal import local_import_closure, dependencies, verify_wire_fields
from extra_token_match import FullPromptCounter, match_padding, ASSETS, FILES
from extra_rescue_design import TASKS, plans, counterfactual
from verify_external_execution_controls_v4 import clocks_only, permitted_clock_paths

ROOT = parent.ROOT
OUT = ROOT/'experiments/results/execution_extension_20261005_v2'
PROTOCOL = ROOT/'docs/paper/execution-extension-protocol-20261005-v2.md'
MODELS = ('glm','qwen')
REPEATS = 2
TURNS = 3
TOKENS = 2400
CAP = 522
N_TASKS = set(TASKS) | {'workspace/user_task_39'}
SYSTEM = parent.SYSTEM + '\nObjects beginning __token_padding are declared neutral NON_EVIDENCE. Ignore them and do not use them as task facts.'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def env_versions():
    return dependencies() | {'token_packages': {n:version(n) for n in ('transformers','tokenizers','jinja2','huggingface-hub')}}

def canonical_arm_map(arm_plans):
    groups={};result={}
    for arm, entry in arm_plans.items():
        key=digest(entry)
        result[arm]=groups.setdefault(key,arm)
    return result

def build_cases():
    data, old=parent.frozen()
    selected=[w for w in data['workflows'] if w['id'] in N_TASKS]
    assert len(selected)==4 and sum(len(w['receivers']) for w in selected)==11
    result=[]
    for w in selected:
        earlier={}
        for role in w['receivers']:
            for dep in role.get('dependencies',[]):
                assert dep in earlier and not earlier[dep]['tools'], 'Extension only admits ordered empty-tool predecessor feedback'
            earlier[role['id']]=role
    for model in MODELS:
        for w in selected:
            for rep in range(REPEATS):
                path=parent.OUT/model/w['id'].replace('/','__')/f'repeat{rep}'/'selection.json'
                selection=read(path)['selected'];assert selection is not None
                union=sorted(set().union(*(set(v) for v in selection.values())))
                n={'D_short':{'visible':selection,'token_budget':False},
                   'D_token':{'visible':selection,'token_budget':True},
                   'U_token':{'visible':{r['id']:union[:] for r in w['receivers']},'token_budget':True}}
                groups=[('N',n)]
                if w['id'] in TASKS:
                    r={name:{'visible':value,'token_budget':True} for name,value in plans(w)['plans'].items()}
                    groups.append(('R',r))
                for group, arm_plans in groups:
                    result.append({'group':group,'workflow':w,'model':model,'repeat':rep,
                         'plans':arm_plans,'canonical':canonical_arm_map(arm_plans),
                         'prior_selection_sha256':sha(path) if group=='N' else None,
                         'target_role':TASKS[w['id']]['target_role'] if group=='R' else None})
    return {'cases':result,'counterfactual_controls':[counterfactual(w) for w in selected if w['id'] in TASKS],
            'parent_freeze_sha256':digest(old),'named_arms':sum(len(c['plans']) for c in result),
            'unique_arms':sum(len(set(c['canonical'].values())) for c in result)}

def prepare():
    data=build_cases()
    assets={}
    for model in MODELS:
        FullPromptCounter(model)
        for name in (*FILES,'manifest.json'):
            path=ASSETS/model/name;assets[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    sources=local_import_closure([Path(__file__)]) | {PROTOCOL}
    manifest={'version':'execution-extension-2','dataset_sha256':digest(data),
              'code_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(sources)},
              'asset_sha256':assets,'dependencies':env_versions(),'source':source_check(),
              'models':list(MODELS),'system':SYSTEM,'request_cap_per_model':CAP,
              'token_budgets':[8192+4096*t for t in range(TURNS)],
              'provider_equality_not_assumed':True,'human_audit_paused':True,
              'post_observation_exploratory_extension':True}
    parent.base.save_once(OUT/'dataset.json',data);parent.base.save_once(OUT/'freeze.json',manifest)
    for name,h in manifest['code_sha256'].items():
        dest=OUT/'frozen_source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==h
        else:dest.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({'prepared':True,'named_arms':data['named_arms'],'unique_arms':data['unique_arms'],'new_calls':0}))

def frozen():
    data,manifest=read(OUT/'dataset.json'),read(OUT/'freeze.json')
    assert digest(data)==manifest['dataset_sha256']
    assert env_versions()==manifest['dependencies'] and source_check()==manifest['source']
    for kind in ('code_sha256','asset_sha256'):
        for name,h in manifest[kind].items():assert sha(ROOT/name)==h,'Frozen source/asset drift: '+name
    return data,manifest

def exclusion(model):
    marker=OUT/model/'model_exclusion.json'
    stops=list((OUT/model).glob('*/api/scope_halted.json'))
    if stops and not marker.exists():
        try:atomic_exclusive(marker,{'exclude_all_effects':True,'reason':read(stops[0])})
        except FileExistsError:pass
    return marker.exists()

class Client(NativeClient):
    def call(self,*args,**kwargs):
        if exclusion(self.model):raise ScopeStopped('entire_extension_model_excluded')
        result=super().call(*args,**kwargs)
        exclusion(self.model)
        return result

def client_for(case):
    # Each task scope is sequential; tasks can run concurrently without sharing caps.
    w=case['workflow'];roles=len(w['receivers'])
    arm_count=9 if w['id'] in TASKS else 3
    return Client(OUT/case['model']/w['id'].replace('/','__')/'api',case['model'],
                  request_cap=REPEATS*roles*arm_count*TURNS,max_tokens=TOKENS)

def case_dir(case):
    return OUT/case['model']/case['workflow']['id'].replace('/','__')/f"repeat{case['repeat']}"/case['group']

def logical(case,arm,rid,turn):
    return f"{case['group']}/{case['workflow']['id']}/repeat{case['repeat']}/{arm}/{rid}/turn{turn}"

def initial_history(w,role,visible,prior):
    dynamic=[{'receiver':dep,'tool_results':[{'function':t['function'],'result':t['result'],'error':t['error']}
              for t in prior[dep]['tool_traces']]} for dep in role.get('dependencies',[])]
    material=[parent.base.visible(x) for x in w['records'] if x['id'] in visible]
    material.append({'id':'__token_padding_0','text':'NON_EVIDENCE ','source':{}})
    user={'original_workflow_request':w['original_prompt'],'assigned_task':role['instruction'],
          'source_visibility_contract':parent.base.RECEIVER,'reference_material':material}
    if dynamic:user['actual_predecessor_tool_results']=dynamic
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(user,ensure_ascii=False)}]

def match_history(counter,history,schema,turn,budget):
    if not budget:return deepcopy(history),{'status':'unmatched_short_control','real_model_calls':0}
    clean=deepcopy(history);payload=json.loads(clean[1]['content'])
    slots=[r for r in payload['reference_material'] if r['id']=='__token_padding_0'];assert len(slots)==1
    slots[0]['text']='NON_EVIDENCE ';clean[1]['content']=json.dumps(payload,ensure_ascii=False)
    return match_padding(counter,None,clean,schema,target_tokens=8192+4096*turn,max_tokens=TOKENS)

def role_run(case,arm,role,loaded,client,counter,prior):
    w=case['workflow'];entry=case['plans'][arm];rid=role['id'];folder=case_dir(case)
    journal=folder/'journal'/arm/rid;schema=loaded.tools_schema(role['tools'])
    history=initial_history(w,role,entry['visible'][rid],prior)
    binding=digest({'case':case,'arm':arm,'role':role,'history':history,'schema':schema,
                    'initial_state':snapshot(loaded.env),'config':client.config})
    row={'id':rid,'visible_ids':entry['visible'][rid],'input':deepcopy(history),'schemas':schema,
         'dynamic_results':json.loads(history[1]['content']).get('actual_predecessor_tool_results',[]),'rounds':[],'text':[],'tool_traces':[],'rejected_calls':[],
         'termination':None,'response_complete':True,'token_matches':[],'actual_request_histories':[],
         'input_delivery_evidence':'not_attempted'}
    seen={}
    for turn in range(TURNS):
        checkpoint=journal/f'turn{turn}.json'
        if checkpoint.exists():
            cp=read(checkpoint);assert cp['binding']==binding
            assert cp['history_before_match']==history
            for response in cp['row']['rounds']:parent.base.receipt_check(response,client)
            parent.base.restore(loaded,cp['state'],cp['state_json'])
            row,history,seen=cp['row'],cp['history'],cp['seen']
            if row['termination']:break
            continue
        before_match=deepcopy(history)
        actual,audit=match_history(counter,history,schema,turn,entry['token_budget'])
        row['token_matches'].append(audit)
        if actual is None:
            row.update(termination='local_match_unavailable',response_complete=False)
        else:
            history=actual
            row['actual_request_histories'].append(deepcopy(history))
            try:
                response=client.call(history,schema,'executor',logical_id=logical(case,arm,rid,turn),max_tokens=TOKENS)
            except (ScopeStopped,IntentOnly) as exc:
                row.update(termination=type(exc).__name__,response_complete=False)
                response=None
            if response is not None:
                row['rounds'].append(response)
                if turn==0:
                    row['input_delivery_evidence']='provider_response_received' if response.get('raw_body') is not None else 'attempted_reception_unknown'
                if response['status']!='ok':
                    row.update(termination=response['status'],response_complete=False)
                else:
                    output=response['parsed'];row['text'].append(output['text']);calls=output['functions']
                    if not calls:row['termination']='model_finished'
                    else:
                        c=calls[0];identity=digest({'name':c['name'],'args':c['args']})
                        if c['id'] in seen and seen[c['id']]['identity']!=identity:
                            row.update(termination='duplicate_call_id_changed_args',response_complete=False)
                        else:
                            if c['id'] in seen:feedback=seen[c['id']]['feedback']
                            elif c['name'] not in role['tools']:
                                feedback={'error':'tool_not_available_for_assigned_role'};row['rejected_calls'].append(c)
                            else:
                                trace=loaded.call(c['name'],c['args'],c['id']);row['tool_traces'].append(trace)
                                feedback={'result':trace['result'],'error':trace['error']}
                            seen[c['id']]={'identity':identity,'feedback':feedback}
                            append_result(history,output,c,parent.base.canonical(feedback),responses=False)
        cp={'binding':binding,'history_before_match':before_match,'history':history,'row':row,'seen':seen,
            'state':snapshot(loaded.env),'state_json':loaded.env.model_dump_json()}
        parent.base.save_once(checkpoint,cp)
        if row['termination']:break
    row['termination']=row['termination'] or 'turn_cap_after_tool';row['final_history']=history
    parent.base.save_once(journal/'complete.json',{'binding':binding,'row':row,'state':snapshot(loaded.env),
                                                   'state_json':loaded.env.model_dump_json()})
    return row

def execute(case,arm,client,counter):
    w=case['workflow'];folder=case_dir(case);path=folder/(arm+'.json')
    loaded,prefix=parent.base.fresh(w);initial=snapshot(loaded.env)
    assert digest(initial)==w['handoff_state_sha256']
    rows=[];prior={}
    for role in w['receivers']:
        row=role_run(case,arm,role,loaded,client,counter,prior)
        rows.append(row);prior[role['id']]=row
    result={'group':case['group'],'workflow':w['id'],'model':case['model'],'repeat':case['repeat'],'arm':arm,
            'plan':case['plans'][arm],'receivers':rows,'handoff_state':initial,'post_state':snapshot(loaded.env),
            'prefix_traces':prefix,'scores':parent.evaluate(w,loaded,rows,prefix),
            'static_relations':sum(len(ids) for ids in case['plans'][arm]['visible'].values()),
            'necessity_gold_used':False,'binding':digest({'case':case,'arm':arm,'config':client.config})}
    parent.base.save_once(path,result)
    return result

def run_task(cases,model,index):
    counter=FullPromptCounter(model);client=client_for(cases[0])
    for case in cases:
        order=list(dict.fromkeys(case['canonical'].values()))
        start=(index+case['repeat'])%len(order);order=order[start:]+order[:start]
        parent.base.save_once(case_dir(case)/'aliases.json',case['canonical'])
        for arm in order:
            if exclusion(model):return
            result=execute(case,arm,client,counter)
            print(json.dumps({'model':model,'task':case['workflow']['id'],'group':case['group'],
                    'repeat':case['repeat'],'arm':arm,'public':result['scores']['public_check']['status'],
                    'native':result['scores']['native_with_prefix']}),flush=True)

def run(model,workers):
    data,freeze=frozen();assert 1<=workers<=2
    if exclusion(model):raise ScopeStopped('entire_extension_model_excluded')
    parent.base.save_once(OUT/model/'plan.json',{'freeze_sha256':digest(freeze),'workers':workers,
        'named_arms':sum(len(c['plans']) for c in data['cases'] if c['model']==model),
        'unique_arms':sum(len(set(c['canonical'].values())) for c in data['cases'] if c['model']==model)})
    groups={}
    for c in data['cases']:
        if c['model']==model:groups.setdefault(c['workflow']['id'],[]).append(c)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_task,cases,model,i) for i,cases in enumerate(groups.values())]
        for f in as_completed(futures):f.result()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--model',choices=MODELS)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        if not a.model:p.error('--model required')
        run(a.model,a.workers)

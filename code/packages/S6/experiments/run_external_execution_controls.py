"""Narrow live execution study, separate from unchanged human-gated protocols."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import run_same_input_independent as base
from same_input_native_api import NativeClient, ScopeStopped, IntentOnly, digest, read, atomic_exclusive
from same_input_formal_checks import check_formal
from run_same_input_formal import local_import_closure, dependencies, verify_wire_fields, role_request_links, bind_request, selector_seed
from workflow_B_runtime import native_utility, snapshot, source_check

ROOT=base.ROOT
OUT=ROOT/'experiments/results/external_execution_controls_20261005_v2'
PROTOCOL=ROOT/'docs/paper/external-execution-controls-protocol-20261005.md'
ARMS=('D','U','P','A','E')
REPEATS=2
MODELS=('glm','qwen')
SYSTEM=base.SYSTEM+'\nObjects whose ID starts with __padding are synthetic non-evidence for a controlled test. Ignore their contents; they provide no task facts.'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compact(x):return json.dumps(x,ensure_ascii=False)
def nbytes(x):return len(compact(x).encode('utf8'))

def make_padding(record,index):
    target=nbytes(base.visible(record))
    item={'id':f'__padding_{index}','text':'NON_EVIDENCE ','source':{}}
    room=target-nbytes(base.visible(item))
    if room<0:raise ValueError('Source object too small for declared filler')
    item['text']+=('0 '*((room+1)//2))[:room]
    assert nbytes(base.visible(item))==target
    return item

def materials(w,selected,arm,rid):
    own=set(selected[rid]);union=set().union(*(set(v) for v in selected.values()))
    records=[x for x in w['records'] if arm=='A' or x['id'] in (own if arm=='D' else union)]
    if arm=='E':records=[]
    if arm=='P':records=[x if x['id'] in own else make_padding(x,i) for i,x in enumerate(records)]
    return records

def execution_workflow(w,selected,arm,rid):
    value=deepcopy(w)
    value['records']=deepcopy(materials(w,selected,arm,rid))
    return value,[x['id'] for x in value['records']]

def configure():
    # Explicit process-local setting; source files and old freezes remain intact.
    base.SYSTEM=SYSTEM

def prepare():
    data,old_manifest=base.frozen()
    assert len(data['workflows'])==18 and sum(len(w['receivers']) for w in data['workflows'])==43
    negative=[]
    for w in data['workflows']:
        loaded,prefix=base.fresh(w)
        rows=[{'id':r['id'],'text':[],'tool_traces':[],'rejected_calls':[],
          'termination':'not_run','response_complete':False} for r in w['receivers']]
        try:native=native_utility(loaded,prefix,'');err=None
        except Exception as exc:native=None;err=type(exc).__name__
        negative.append({'workflow':w['id'],'native_prefix_only':native,'native_error':err,
          'public_check':check_formal(w,loaded,rows,prefix,native,err)})
        all_ids=[x['id'] for x in w['records']]
        selected={r['id']:(all_ids if i==0 else []) for i,r in enumerate(w['receivers'])}
        for r in w['receivers']:
            p=materials(w,selected,'P',r['id']);u=materials(w,selected,'U',r['id'])
            assert nbytes([base.visible(x) for x in p])==nbytes([base.visible(x) for x in u])
            for padded,original in zip(p,u):
                assert nbytes(base.visible(padded))==nbytes(base.visible(original))
                if original['id'] not in selected[r['id']]:
                    assert padded['source']=={} and padded['id'].startswith('__padding')
                    assert padded['text'].startswith('NON_EVIDENCE ') and set(padded['text'][13:])<={'0',' '}
        assert digest(snapshot(loaded.env))==w['handoff_state_sha256']
    files=local_import_closure([Path(__file__),ROOT/'experiments/analyze_external_execution_controls.py'])
    files.add(PROTOCOL)
    manifest={'version':'external-execution-controls-1','dataset_sha256':digest(data),
      'parent_freeze_sha256':digest(old_manifest),'source':source_check(),
      'models':list(MODELS),'arms':list(ARMS),'repetitions':REPEATS,'workflow_ids':[w['id'] for w in data['workflows']],
      'request_cap_per_model':REPEATS*(18+43*len(ARMS)*base.TURNS),
      'code_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(files)},
      'dependencies':dependencies(),'system':SYSTEM,'padding':'whole non-own object replaced, exact UTF8 bytes; not token matched',
      'human_necessity_gold':False,'old_human_gate_unchanged':True,'new_response_only':True,
      'models_excluded_on_any_identity_anomaly':True}
    base.save_once(OUT/'dataset.json',data)
    base.save_once(OUT/'freeze.json',manifest)
    base.save_once(OUT/'prefix_negative_controls.json',negative)
    for rel in manifest['code_sha256']:
        target=OUT/'frozen_source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==manifest['code_sha256'][rel]
        else:target.write_bytes((ROOT/rel).read_bytes())
    print(json.dumps({'prepared':True,'planned_workflow_arms':18*REPEATS*len(ARMS)*len(MODELS),
      'request_cap_per_model':manifest['request_cap_per_model'],'prefix_native_true':sum(x['native_prefix_only'] is True for x in negative),'new_calls':0}))

def frozen():
    data,manifest=read(OUT/'dataset.json'),read(OUT/'freeze.json')
    assert digest(data)==manifest['dataset_sha256']
    assert source_check()==manifest['source']
    assert dependencies()==manifest['dependencies'],'Execution environment drift'
    for rel,h in manifest['code_sha256'].items():assert sha(ROOT/rel)==h,'Source drift: '+rel
    configure()
    return data,manifest

def refresh_exclusion(model):
    marker=OUT/model/'model_exclusion.json'
    halted=list((OUT/model).glob('*/api/scope_halted.json'))
    if halted and not marker.exists():
        try:atomic_exclusive(marker,{'exclude_all_effects':True,'scope':str(halted[0].parent),
          'reason':read(halted[0])})
        except FileExistsError:pass
    return marker.exists()

class StudyClient(NativeClient):
    def call(self,*args,**kwargs):
        marker=OUT/self.model/'model_exclusion.json'
        if refresh_exclusion(self.model):raise ScopeStopped('entire_model_scope_excluded')
        value=super().call(*args,**kwargs)
        if (self.root/'scope_halted.json').exists():
            try:atomic_exclusive(marker,{'exclude_all_effects':True,'scope':str(self.root),
              'reason':read(self.root/'scope_halted.json')})
            except FileExistsError:pass
        return value

def client_for(w,model):
    return StudyClient(OUT/model/w['id'].replace('/','__')/'api',model,
      request_cap=REPEATS*(1+len(w['receivers'])*len(ARMS)*base.TURNS),max_tokens=base.TOKENS)

def evaluate(w,loaded,rows,prefix):
    flat=[t for r in rows for t in r['tool_traces']]
    text='\n'.join(t for r in rows for t in r['text'] if t)
    values={}
    for label,traces in (('native_with_prefix',prefix+flat),('native_executor_only',flat)):
        try:values[label]=native_utility(loaded,traces,text);values[label+'_error']=None
        except Exception as exc:values[label]=None;values[label+'_error']=type(exc).__name__
    values['public_check']=check_formal(w,loaded,rows,prefix,values['native_with_prefix'],values['native_with_prefix_error'])
    return values

def execute(w,selected,client,repeat,arm,folder,replay=False):
    path=folder/(arm+'.json')
    binding=digest({'workflow':w,'model':client.config,'repeat':repeat,'arm':arm,'selected':selected,'system':SYSTEM})
    if path.exists() and not replay:
        value=read(path);assert value['binding']==binding
        return value
    loaded,prefix=base.fresh(w);initial=snapshot(loaded.env)
    assert digest(initial)==w['handoff_state_sha256']
    prior={};input_static={};real_ids={}
    for role in w['receivers']:
        rid=role['id'];modified,ids=execution_workflow(w,selected,arm,rid)
        if replay:
            assert (folder/'journal'/arm/rid/'complete.json').exists(),'Missing complete journal'
        prior[rid]=base.run_node(modified,role,ids,loaded,prior,client,repeat,arm,folder)
        input_static[rid]=[base.visible(x) for x in modified['records']]
        real_ids[rid]=[x['id'] for x in modified['records'] if not x['id'].startswith('__padding')]
    from deepdiff import DeepDiff
    value={'workflow':w['id'],'model':client.model,'repeat':repeat,'arm':arm,'binding':binding,
      'selection':selected,'static_material':input_static,'real_source_ids':real_ids,
      'prefix_traces':prefix,'handoff_state':initial,'post_state':snapshot(loaded.env),
      'full_state_delta':json.loads(DeepDiff(initial,snapshot(loaded.env),ignore_order=False).to_json()),
      'receivers':list(prior.values()),'scores':evaluate(w,loaded,list(prior.values()),prefix),
      'necessity_gold_used':False,'static_relations':sum(map(len,real_ids.values()))}
    if replay:
        assert value==read(path),'Replayed arm differs'
    else:base.save_once(path,value)
    return value

def run_task(w,model,index):
    client=client_for(w,model)
    for repeat in range(REPEATS):
        folder=OUT/model/w['id'].replace('/','__')/f'repeat{repeat}'
        try:
            selection=base.select(w,client,repeat,folder)
            if selection['selected'] is None:
                base.save_once(folder/'selection_failure.json',{'status':selection['response']['status'],'errors':selection['errors']})
                print(json.dumps({'model':model,'task':w['id'],'repeat':repeat,'selection_failure':True}),flush=True)
                continue
            start=(index+repeat)%len(ARMS)
            for arm in ARMS[start:]+ARMS[:start]:
                if (OUT/model/'model_exclusion.json').exists():return
                value=execute(w,selection['selected'],client,repeat,arm,folder)
                print(json.dumps({'model':model,'task':w['id'],'repeat':repeat,'arm':arm,
                  'public':value['scores']['public_check']['status'],'native':value['scores']['native_with_prefix'],
                  'state':value['scores']['public_check']['full_state_passed']}),flush=True)
        except (ScopeStopped,IntentOnly) as exc:
            base.save_once(folder/'unfinished.json',{'status':type(exc).__name__,'reason':str(exc)})
            print(json.dumps({'model':model,'task':w['id'],'repeat':repeat,'unfinished':type(exc).__name__}),flush=True)
            return

def run(model,workers):
    data,manifest=frozen()
    assert model in MODELS and 1<=workers<=2
    if refresh_exclusion(model):raise ScopeStopped('entire_model_scope_excluded')
    base.save_once(OUT/model/'plan.json',{'freeze_sha256':digest(manifest),'workflow_ids':manifest['workflow_ids'],
      'arms':list(ARMS),'repeats':REPEATS,'planned_arms':180,'parallel_tasks':workers})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(run_task,w,model,i) for i,w in enumerate(data['workflows'])]
        for job in as_completed(jobs):job.result()

def verify():
    data,manifest=frozen();count=0;receipts=0
    for model in MODELS:
        for w in data['workflows']:
            root=OUT/model/w['id'].replace('/','__')
            if not (root/'api/config.json').exists():continue
            client=client_for(w,model)
            for path in (root/'api/responses').glob('*.json'):
                response=read(path);base.receipt_check(response,client);verify_wire_fields(response,model);receipts+=1
            # Replay uses complete journals and cached receipts only.
            def never_send(*args,**kwargs):raise AssertionError('Verifier attempted a network request')
            client.call=never_send
            for repeat in range(REPEATS):
                folder=root/f'repeat{repeat}'
                if not (folder/'selection.json').exists():continue
                selection=base.select(w,client,repeat,folder)
                history,schema=selector_seed(w)
                linked=set();bind_request(client,f'{w["id"]}/repeat{repeat}/selector','selector',history,schema,linked)
                if selection['selected'] is None:continue
                for arm in ARMS:
                    if (folder/(arm+'.json')).exists():
                        execute(w,selection['selected'],client,repeat,arm,folder,True);count+=1
                        loaded,_=base.fresh(w);prior={}
                        for role in w['receivers']:
                            modified,ids=execution_workflow(w,selection['selected'],arm,role['id'])
                            role_request_links(modified,role,ids,loaded,prior,client,repeat,arm,folder,linked)
                            prior[role['id']]=base.run_node(modified,role,ids,loaded,prior,client,repeat,arm,folder)
    print(json.dumps({'offline_replay_passed':True,'arms':count,'receipts':receipts,'new_calls':0}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify'])
    parser.add_argument('--model',choices=MODELS);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='verify':verify()
    else:
        if not args.model:parser.error('--model required')
        run(args.model,args.workers)

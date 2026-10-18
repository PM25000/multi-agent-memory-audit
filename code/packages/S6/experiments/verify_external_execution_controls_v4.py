"""Offline verification amendment; never alters frozen execution or receipts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import run_external_execution_controls as study
import run_same_input_formal as formal
from same_input_native_api import read, digest, encoded
from workflow_B_api import build_request
from workflow_B_runtime import snapshot, digest as state_digest

def bind_request(client,logical,phase,history,schema,linked):
    key=hashlib.sha256(logical.encode('utf8')).hexdigest()+'.json'
    request=build_request(client.spec,history,schema,max_tokens=study.base.TOKENS)
    expected={'logical_id':logical,'phase':phase,'model_key':client.model,'request':request,
      'request_sha256':digest(request),'request_bytes':len(encoded(request)),
      'scope_configuration_sha256':digest(client.config)}
    for group in ('intents','preflight_rejections'):
        path=client.root/group/key
        if path.exists():
            value=read(path)
            assert all(value.get(k)==v for k,v in expected.items()),'Actual public request differs'
            linked.add((group,key))
            receipt=client.root/'responses'/key
            if receipt.exists():
                value=read(receipt)
                # Phase is an intent fact, absent from native provider receipts.
                assert all(value.get(k)==expected[k] for k in ('logical_id','model_key','request_sha256'))
                linked.add(('responses',key))

def permitted_clock_paths(before,post,function,args):
    paths=set();result_paths=set()
    if function in ('send_email','create_calendar_event'):
        old=before.get('inbox',{}).get('emails',{})
        new=post.get('inbox',{}).get('emails',{})
        added=set(new)-set(old)
        for key in added:paths.add(f"root['inbox']['emails'][{key!r}]['timestamp']")
        for group in ('sent','received','drafts'):
            for index,item in enumerate(post.get('inbox',{}).get(group,[])):
                if item['id_'] in added:paths.add(f"root['inbox'][{group!r}][{index}]['timestamp']")
        if function=='send_email' and len(added)==1:result_paths.add("root['timestamp']")
    if function in ('append_to_file','create_file'):
        old=before.get('cloud_drive',{}).get('files',{})
        new=post.get('cloud_drive',{}).get('files',{})
        touched={str(args['file_id'])} if function=='append_to_file' else set(new)-set(old)
        for key in touched:
            if key in new:paths.add(f"root['cloud_drive']['files'][{key!r}]['last_modified']")
        for index,item in enumerate(post.get('cloud_drive',{}).get('initial_files',[])):
            if item['id_'] in touched:paths.add(f"root['cloud_drive']['initial_files'][{index}]['last_modified']")
        if len(touched)==1:result_paths.add("root['last_modified']")
    return paths,result_paths

def clocks_only(expected,actual,permitted=()):
    from deepdiff import DeepDiff
    delta=json.loads(DeepDiff(expected,actual,ignore_order=False).to_json())
    allowed=[]
    if delta:
        if set(delta)!={'values_changed'}:return False,[]
        for path in delta['values_changed']:
            if path not in permitted:return False,[]
            allowed.append(path)
    return True,allowed

def replay_tools(w,model,rep,arm,folder):
    loaded,_=study.base.fresh(w);tool_count=0;clock_paths=[]
    for role in w['receivers']:
        prior_traces=[]
        for turn in range(study.base.TURNS):
            path=folder/'journal'/arm/role['id']/f'turn{turn}.json'
            if not path.exists():break
            cp=read(path);traces=cp['row']['tool_traces']
            added=traces[len(prior_traces):]
            assert len(added)<=1
            if added:
                trace=added[0]
                before=snapshot(loaded.env)
                assert trace['before_sha256']==state_digest(before)
                new=loaded.call(trace['function'],trace['args'],trace['id'])
                assert new['error']==trace['error']
                # Result clocks may vary only when the corresponding world differs
                # at an explicitly identified native clock field.
                permitted,result_paths=permitted_clock_paths(before,cp['state'],trace['function'],trace['args'])
                equal,paths=clocks_only(cp['state'],snapshot(loaded.env),permitted)
                assert equal,'Actual native tool replay state differs'
                if new['result']!=trace['result']:
                    same,_=clocks_only(trace['result'],new['result'],result_paths)
                    assert paths and same,'Tool result differs beyond touched object native clock'
                clock_paths.extend(paths);tool_count+=1
            else:assert snapshot(loaded.env)==cp['state'],'State changed without action'
            assert state_digest(cp['state'])==(added[-1]['after_sha256'] if added else state_digest(snapshot(loaded.env)))
            study.base.restore(loaded,cp['state'],cp['state_json'])
            prior_traces=traces
            if cp['row']['termination']:break
    saved=read(folder/(arm+'.json'))
    assert snapshot(loaded.env)==saved['post_state']
    return tool_count,clock_paths

def verify(write=False):
    data,freeze=study.frozen()
    original_selector,original_bind,formal_bind=study.selector_seed,study.bind_request,formal.bind_request
    try:
        def selector_history_schema(w):
            info,schema,history=original_selector(w)
            return history,schema
        study.selector_seed=selector_history_schema
        study.bind_request=bind_request;formal.bind_request=bind_request
        study.verify()
    finally:
        study.selector_seed=original_selector;study.bind_request=original_bind;formal.bind_request=formal_bind
    tools=0;arms=0;clocks=[];hashes={}
    for model in study.MODELS:
        for w in data['workflows']:
            root=study.OUT/model/w['id'].replace('/','__')
            for rep in range(study.REPEATS):
                folder=root/f'repeat{rep}'
                for arm in study.ARMS:
                    path=folder/(arm+'.json')
                    if path.exists():
                        n,paths=replay_tools(w,model,rep,arm,folder);tools+=n;clocks+=paths;arms+=1
                        hashes[str(path.relative_to(study.ROOT)).replace('\\','/')]=study.sha(path)
    result={'offline_verification_passed':True,'workflow_arms':arms,'actual_native_tool_replays':tools,
      'new_model_requests':0,'freeze_sha256':digest(freeze),'verified_arm_sha256':hashes,
      'clock_exception_paths':sorted(set(clocks)),
      'clock_limit':'Only clock paths of objects actually created or touched by this native call may differ; old untouched object clocks are exact; wall-clock authenticity is not verified',
      'verifier_source_sha256':study.sha(Path(__file__)),
      'amendment':'selector_seed returns info/schema/history; native receipt phase is verified through request intent'}
    if write:(study.OUT/'verification_v4.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('verified_arm_sha256','clock_exception_paths')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();verify(args.write)

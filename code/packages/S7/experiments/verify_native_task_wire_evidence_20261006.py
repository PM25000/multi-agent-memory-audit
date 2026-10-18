"""Independent stdlib-only saved-evidence check; never imports experiment runtime/API."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/results/native_task_execution_20261006'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def request_hash(v):return hashlib.sha256((json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()).hexdigest()
def records(content,pool):
    try:value=json.loads(content)
    except (ValueError,TypeError):return set()
    if not isinstance(value,dict) or not isinstance(value.get('records'),list):return set()
    found=set()
    for row in value['records']:
        if isinstance(row,dict) and row.get('id') in pool and row==pool[row['id']]:found.add(row['id'])
    return found
def main():
    freeze=read(DEST/'freeze.json');data=read(DEST/'dataset.json');summary=read(DEST/'summary.json')
    assert sha(DEST/'dataset.json')==freeze['dataset_sha256']
    from public_checks import original_subset
    original_subset(freeze['files'])
    roles=0;receptions=0;receipts=0;provider_ids={};independent={}
    for model in ('qwen','glm'):
        base=DEST/'models'/model;config=read(base/'api/config.json');wire={};intent={};ids=set()
        assert not config['injected_offline_transport'] and config['sdk_retries']==0
        assert not (base/'api/scope_halted.json').exists()
        for p in (base/'api/responses').glob('*.json'):
            response=read(p);planned=read(base/'api/intents'/p.name)
            assert request_hash(planned['request'])==response['request_sha256']==planned['request_sha256']
            assert response['api_attempted'] and response['status']=='ok' and response['http_status']==200
            raw=json.loads(response['raw_body']);assert raw==response['raw_response']
            assert raw['model'] in config['accepted_response_models'];ids.add(raw['model'])
            assert len(raw['choices'])==1 and raw['choices'][0]['message']['role']=='assistant'
            assert response['logical_id']==planned['logical_id']
            wire[response['logical_id']]=response;intent[response['logical_id']]=planned['request'];receipts+=1
        recomputed={a:{'received_sources':0,'received_outside_A':0,'http_sources':0,'http_outside_A':0,'roles_http_outside_A':0} for a in ('U','D','C','F')}
        for w in data['workflows']:
            selector=read(base/'cases'/w['id']/'selection.json')['selection'];assert selector is not None
            selected_raw=wire[w['id']+'/select']['raw_response']['choices'][0]['message']['tool_calls']
            assert len(selected_raw)==1 and selected_raw[0]['function']['name']=='select_records'
            assert json.loads(selected_raw[0]['function']['arguments'])==selector
            for arm in recomputed:
                t=read(base/'cases'/w['id']/(arm+'.json'));pub={p['message_id']:p for p in t['publications']}
                assert len(pub)==len(t['publications'])
                for role in ('role_a','role_b'):
                    actual=set()
                    for e in t['receptions']:
                        if e['role']!=role or e['phase']!='handoff':continue
                        assert e['message_id'] in pub and e['message']==pub[e['message_id']]['message']
                        assert e['sender']==pub[e['message_id']]['sender'] and e['topic']==pub[e['message_id']]['topic']
                        actual|=records(e['message']['body']['content'],w['sources']);receptions+=1
                    c=next(x for x in t['client_calls'] if x['role']==role)
                    logical=w['id']+'/'+arm+'/'+role+'/round0';assert c['requests'][0]['logical_id']==logical
                    sent=intent[logical]['messages'];assert sent==c['first_http_messages']
                    transmitted=set()
                    for m in sent:
                        if m['role']=='user':transmitted|=records(m['content'],w['sources'])
                    assert transmitted==actual
                    s=recomputed[arm];s['received_sources']+=len(actual);s['received_outside_A']+=len(actual-set(w['allowed'][role]))
                    s['http_sources']+=len(transmitted);s['http_outside_A']+=len(transmitted-set(w['allowed'][role]))
                    s['roles_http_outside_A']+=bool(transmitted-set(w['allowed'][role]));roles+=1
                for p in t['publications']:
                    if p['phase']!='handoff' or p['sender'] is not None:continue
                    envelope=json.loads(p['message']['body']['content']);role=envelope['intended_role']
                    expected=set(selector[role])
                    if arm=='C':expected&=set(w['allowed'][role])
                    assert records(p['message']['body']['content'],w['sources'])==expected
                    assert p['topic']==(role if arm=='D' else 'native_task_common')+'/default'
        for arm,counts in recomputed.items():
            for k,v in counts.items():assert summary[model]['arms'][arm][k]==v,(model,arm,k,v)
        provider_ids[model]=sorted(ids);independent[model]=recomputed
    assert roles==192
    output={'passed':True,'role_arm_positions':roles,'handoff_receptions_checked':receptions,
        'actual_provider_receipts':receipts,'returned_models':provider_ids,'counts':independent,
        'method':'stdlib-only raw JSON+native message ID/body/topic/sender+actual SDK request argument check; no experiment imports or API calls',
        'public_projection_verified':True,'original_manuscript_gate_replayed':False}
    expected=read(DEST/'independent_wire_verification.json')
    expected.pop('source_and_manuscript_hashes_unchanged',None)
    expected.update(public_projection_verified=True,original_manuscript_gate_replayed=False)
    assert output==expected
    print(json.dumps({k:v for k,v in output.items() if k!='counts'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

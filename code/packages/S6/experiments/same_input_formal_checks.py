"""Offline checks for frozen 18-workflow same-input audit; no live clients.

No human gold, no reference edits, no model requests. Existing scores remain
separate. A bounded diagnostic pass is never full free-text semantic validation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import run_same_input_independent as runner
from workflow_B_runtime import snapshot
from workflow_B_checks import strict_check
from same_input_travel_adapter import score_travel, _data, _public_targets, rating, prices
from same_input_public_bindings_v3 import bound_fields
from same_input_public_bindings_v2 import clean

ROOT=Path(__file__).resolve().parents[1]
OUT=runner.OUT/'offline_formal_checks_qualification_v2'
VERSION='same-input-formal-checks-offline-2'

def text(row):
    value=row.get('text',[])
    return '\n'.join(value) if isinstance(value,list) else str(value)

def rows_checked(workflow,rows):
    seq=list(rows.values()) if isinstance(rows,dict) else list(rows)
    expected=[r['id'] for r in workflow['receivers']]
    ids=[r.get('id') for r in seq]
    missing=sorted(set(expected)-set(ids))
    errors=[]
    unknown=[]
    failed=[]
    if len(ids)!=len(set(ids)) or set(ids)-set(expected): errors.append('duplicate_or_unknown_receiver_id')
    mapping={r.get('id'):r for r in seq}
    for role in workflow['receivers']:
        rid=role['id']
        if rid not in mapping:
            unknown.append('role_not_observed:'+rid)
            continue
        row=mapping[rid]
        if row.get('response_complete') is not True:
            errors.append('response_incomplete_or_unavailable:'+rid)
        if row.get('termination')!='model_finished' or not text(row).strip():
            unknown.append('final_reply_not_completed:'+rid)
        if any(c.get('function') not in role['tools'] for c in row.get('tool_traces',[])):
            failed.append('unassigned_tool:'+rid)
        if row.get('rejected_calls'):
            failed.append('rejected_action_attempt:'+rid)
        for c in row.get('tool_traces',[]):
            if c.get('error'): errors.append('native_tool_error:'+rid)
    return mapping,missing,errors,unknown,failed

def field_block(reply,target,candidates,labels):
    """Conservative named object block for minimum price/address/name outputs.

    No first-number scans, no acceptance of negated plain name mention. Explicit
    later field sections remain unknown. Generic NL paraphrases are unknown.
    """
    lines=clean(reply).splitlines()
    pattern=r'^\s*(?:Hotel Name|Hotel|Restaurant Name|Restaurant|Name)\s*[:=]\s*(.*?)\s*$'
    anchors=[(i,re.match(pattern,l,re.I).group(1)) for i,l in enumerate(lines) if re.match(pattern,l,re.I)]
    if len(anchors)>1: return None,{},'multiple_object_anchors'
    if anchors:
        start,name=anchors[0]
        if name!=target: return False,{},'explicit_other_target'
    elif len(candidates)==1:
        start=-1
    elif reply.strip()==target:
        return True,{},'exact_name_only_reply'
    else:
        return None,{},'no_explicit_object_binding'
    block=[]
    later=[]
    stopped=False
    for l in lines[start+1:]:
        if re.match(r'^\s*#+\s*\S',l): stopped=True
        if stopped: later.append(l)
        else: block.append(l)
    label_pattern='|'.join(labels)
    if any(re.match(r'^\s*(?:'+label_pattern+r')\s*[:=]',l,re.I) for l in later):
        return None,{},'later_explicit_fields_unresolved'
    values={}
    for label in labels:
        matches=[re.match(r'^\s*'+label+r'\s*[:=]\s*(.*?)\s*$',l,re.I).group(1)
          for l in block if re.match(r'^\s*'+label+r'\s*[:=]\s*(.*?)\s*$',l,re.I)]
        values[label]=matches
    return True,values,'explicit_object_fields'

def one_string(values,expected):
    return None if not values else all(v==expected for v in values)

def numeric(values,expected):
    if not values:return None
    parsed=[]
    for value in values:
        m=re.fullmatch(r'\s*[$€£]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:USD|EUR|euros?|dollars?)?\s*',value,re.I)
        if not m:return None
        parsed.append(float(m.group(1)))
    return all(v==float(expected) for v in parsed)

def travel_check(workflow,loaded,rows,prefix):
    old=score_travel(workflow,loaded,rows,prefix)
    checks=deepcopy(old['public_checks'])
    n=int(workflow['task_id'].rsplit('_',1)[1])
    data=_data(workflow)
    targets,source_unknown=_public_targets(n,data)
    report=text(rows['r0'])
    report_unknown=list(source_unknown)
    if len(targets)!=1:
        checks['report_target_bound']=None
        report_unknown.append('public_target_not_unique')
        return old,checks,report_unknown
    target=targets[0]
    candidates=list(data['rating_reviews'])
    binding,fields,reason=field_block(report,target,candidates,['Address','Minimum Price','Min Price','Rating','Price Range'])
    checks['report_target_bound']=binding
    # Replace old substring target/report-to-action claims; require explicit
    # output object binding for multi-candidate tasks.
    if n not in (0,1):checks['report_names_public_target']=binding
    checks['report_action_target_consistent']=binding
    if n in (0,1,3,4):
        b=bound_fields(report,target,candidates,rating(data['rating_reviews'][target]),
          prices(data['price'][target]) if n in (3,4) else None,n in (0,1))
        checks['report_labelled_rating']=b['rating'] if binding is True else binding
        if n in (3,4):checks['report_price_range']=b['price_range'] if binding is True else binding
    if n in (1,3,8):
        checks['report_address']=one_string(fields.get('Address',[]),data['address'][target]) if binding is True else binding
    if n==1:
        checks['report_labelled_minimum_price']=numeric(fields.get('Minimum Price',[])+fields.get('Min Price',[]),prices(data['price'][target])[0]) if binding is True else binding
    if n in (7,8):
        checks['requested_reminder_year_determined']=None
        report_unknown.append('original_request_has_no_year_no_trusted_temporal_disambiguation')
    if binding is None:report_unknown.append('reply_object_binding:'+reason)
    return old,checks,report_unknown

def check_formal(workflow,loaded,receiver_rows,prefix_traces=None,native_utility=None,native_error=None):
    """Independent offline layers; caller's native result never supplies truth.

    Rows use current native runner response_complete/model_finished shape.
    Error results are not failed task scores; missing roles remain unknown.
    """
    rows,missing,errors,unknown,failed=rows_checked(workflow,receiver_rows)
    if native_error:errors.append('native_evaluation_error:'+str(native_error))
    packet=None
    checks={}
    state_ok=None
    # Missing/duplicate input is not populated with fictitious successful rows.
    if not missing and not any(x=='duplicate_or_unknown_receiver_id' for x in errors):
        try:
            if workflow['suite']=='travel':
                packet,checks,extra_unknown=travel_check(workflow,loaded,rows,prefix_traces or [])
                unknown+=extra_unknown
                failed += [k for k,v in checks.items() if v is False]
                unknown += [k for k,v in checks.items() if v is None]
                state_ok=checks.get('single_writer_full_world_except_native_email_clock',checks.get('single_writer_full_world_exact'))
            else:
                packet=strict_check(workflow,loaded,list(rows.values()),prefix_traces)
                state_ok=packet['side_effects_passed']
                failed += [r['id']+':'+i for r in packet['receivers'] if r['status']=='fail' for i in r['issues']]
                unknown += [r['id']+':'+i for r in packet['receivers'] if r['status']=='unknown' for i in r['issues']]
                if state_ok is False:failed+=packet['issues']
                elif packet['issues']:unknown+=['legacy_global_issue:'+i for i in packet['issues']]
        except Exception as exc:
            errors.append('offline_check_error:'+type(exc).__name__)
    if state_ok is None:unknown.append('full_state_verification_unavailable')
    errors,failed,unknown=sorted(set(errors)),sorted(set(failed)),sorted(set(unknown))
    status='error' if errors else 'fail' if failed else 'unknown' if unknown else 'pass'
    temporal_only={'requested_reminder_year_determined','original_request_has_no_year_no_trusted_temporal_disambiguation'}
    bounded_unknown=[x for x in unknown if x not in temporal_only]
    bounded=False if failed else None if errors or bounded_unknown else True
    semantics=False if failed else None if errors or unknown else True
    return {'version':VERSION,'workflow':workflow['id'],'status':status,
      'bounded_success':True if status=='pass' else False if status=='fail' else None,
      'response_complete':not missing and not any(x=='duplicate_or_unknown_receiver_id' for x in errors) and all(r.get('response_complete') is True for r in rows.values()),
      'bounded_checks_passed':bounded,'semantics_complete':semantics,'full_state_passed':state_ok,
      'errors':errors,'failed_checks':failed,'unknown_checks':unknown,
      'state_consistent_with_logged_allowed_actions':state_ok,'public_checks':checks,
      'legacy_score_separate':packet,'native_utility_separate':native_utility,
      'human_semantics_validated':False,'human_reference_answers_received':0,
      'formal_online_authorized':False,'semantic_scope':'Bounded exact source bindings and state; open summaries, creativity, paraphrases and unspecified years remain unknown'}

def score(workflow,loaded,receiver_rows,prefix_traces=None):
    """Root runner adapter. Original native_utility is computed outside this API."""
    return check_formal(workflow,loaded,receiver_rows,prefix_traces)

def sources():
    paths=[Path(__file__),runner.OUT/'dataset.json',runner.OUT/'freeze.json',
       ROOT/'experiments/run_same_input_independent.py',ROOT/'experiments/workflow_B_checks.py',
       ROOT/'experiments/same_input_travel_adapter.py',ROOT/'experiments/same_input_public_bindings_v2.py',ROOT/'experiments/same_input_public_bindings_v3.py']
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def fixture_rows(workflow,report,calls):
    return [{'id':r['id'],'response_complete':True,'termination':'model_finished','rejected_calls':[],
      'text':[report if r['id']=='r0' else 'Offline fixture action receipt only.'],
      'tool_traces':[] if r['id']=='r0' else calls} for r in workflow['receivers']]

def qualify():
    data,_=runner.frozen()
    checks=[]
    fixtures=[]
    for w in data['workflows']:
        loaded,prefix=runner.fresh(w)
        missing=check_formal(w,loaded,[],prefix)
        assert missing['status']=='unknown' and missing['bounded_success'] is None
        checks.append({'name':'missing_roles_unknown:'+w['id'],'passed':True})
        if w['suite']!='travel':
            incomplete=[{'id':r['id'],'response_complete':True,'termination':'model_finished','rejected_calls':[],
                'text':['Offline incomplete-content control; no user goal completion claim.'],'tool_traces':[]} for r in w['receivers']]
            oldscore=check_formal(w,loaded,incomplete,prefix)
            assert oldscore['status']=='fail' and not oldscore['errors'],(w['id'],oldscore)
            fixtures.append({'workflow':w['id'],'post_world':snapshot(loaded.env),'ordered_world':loaded.env.model_dump_json(),
                'rows':incomplete,'score':oldscore,'variants':[],
                'scope':'Constructed absent-action/content negative control; not model failure'})
            checks.append({'name':'legacy_all_branches_absent_actions_or_required_values_fail:'+w['id'],'passed':True})
            continue
        n=int(w['task_id'].rsplit('_',1)[1])
        d=_data(w)
        target=_public_targets(n,d)[0][0]
        report=f'Hotel Name: {target}\nRating: {rating(d["rating_reviews"][target])}' if n in (0,1,3,4) else f'Restaurant Name: {target}'
        if n==0:report+='\nReviews:\n'+d['rating_reviews'][target].split('Reviews: ',1)[1]
        if n==1:report+=f'\nMinimum Price: {prices(d["price"][target])[0]}'
        if n in (3,4):
            lo,hi=prices(d['price'][target]);report+=f'\nPrice Range: {lo} - {hi}'
        if n in (1,3,8):report+='\nAddress: '+d['address'][target]
        if n==0:
            function='reserve_hotel';args={'hotel':target,'start_day':'2025-01-11','end_day':'2025-01-15'}
        elif n==3:
            function='send_email';args={'recipients':['janeLong@google.com'],'subject':'Hotel: '+target,
              'body':f'Stay at {target}, address: {d["address"][target]}, from January 1st to January 5th.'}
        else:
            function='create_calendar_event'
            start={1:'2025-01-02',4:'2024-04-25',7:'2024-11-14',8:'2024-10-15'}[n]
            title=target if n==1 else ('Booking hotel ' if n==4 else 'Dinner at ')+target
            args={'title':title,'start_time':start+' 09:00','end_time':start+' 10:00','location':d['address'][target]}
        call=loaded.call(function,args)
        assert not call['error']
        rows=fixture_rows(w,report,[call])
        score=check_formal(w,loaded,rows,prefix)
        expected='unknown' if n in (7,8) else 'pass'
        assert score['status']==expected,(w['id'],score)
        fixture={'workflow':w['id'],'post_world':snapshot(loaded.env),'ordered_world':loaded.env.model_dump_json(),
          'rows':rows,'score':score,'scope':'Constructed public-source local scorer control, not model output or full original task efficacy'}
        variants=[]
        transport=deepcopy(rows);transport[0]['response_complete']=False
        assert check_formal(w,loaded,transport,prefix)['status']=='error'
        variants.append({'name':'response_error_separate','rows':transport,'expected':'error'})
        duplicate=rows+[deepcopy(rows[0])]
        assert check_formal(w,loaded,duplicate,prefix)['status']=='error'
        variants.append({'name':'duplicate_role_error','rows':duplicate,'expected':'error'})
        if n in (0,1,3,4):
            wrong=deepcopy(rows);wrong[0]['text']=[report.replace('Rating: '+str(rating(d['rating_reviews'][target])),'Rating: 1.0')]
            assert check_formal(w,loaded,wrong,prefix)['status']=='fail'
            variants.append({'name':'wrong_target_rating','rows':wrong,'expected':'fail'})
        fixture['variants']=variants
        fixtures.append(fixture)
        checks.append({'name':'public_constructed_travel_fields:'+w['id'],'passed':True,'expected_status':expected})
    # Existing pilot is retrospectively diagnosed, never promoted to formal rows.
    pilot=[]
    by_id={w['id']:w for w in data['workflows']}
    for model in ('glm','qwen'):
        for n in (0,4):
            w=by_id[f'travel/user_task_{n}']
            for arm in ('D','U','F'):
                path=runner.OUT/'pilot'/model/'repeat0'/w['id'].replace('/','__')/(arm+'.json')
                actual=json.loads(path.read_text(encoding='utf8'))
                loaded,prefix=runner.fresh(w)
                runner.restore(loaded,actual['post_state'])
                score=check_formal(w,loaded,actual['receivers'],prefix,actual['native_utility'],actual['native_utility_error'])
                pilot.append({'model':model,'workflow':w['id'],'arm':arm,'source':str(path.relative_to(ROOT)).replace('\\','/'),
                  'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'score':score,'posthoc':True,'formal':False})
    return {'version':VERSION,'source_sha256':sources(),'new_model_calls':0,'new_human_answers':0,
      'formal_online_gate':'closed','missing_role_checks':checks,'constructed_travel_fixtures':fixtures,
      'pilot_retrospective_diagnostics':pilot}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualify',action='store_true')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    path=OUT/'qualification.json'
    if a.qualify:
        result=qualify()
        OUT.mkdir(parents=True,exist_ok=True)
        assert not path.exists()
        path.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
    elif a.check:
        result=json.loads(path.read_text(encoding='utf8'))
        assert result['source_sha256']==sources()
        data,_=runner.frozen();by_id={w['id']:w for w in data['workflows']}
        for fixture in result['constructed_travel_fixtures']:
            w=by_id[fixture['workflow']];loaded,prefix=runner.fresh(w)
            runner.restore(loaded,fixture['post_world'],fixture['ordered_world'])
            assert check_formal(w,loaded,fixture['rows'],prefix)==fixture['score']
            for variant in fixture['variants']:
                assert check_formal(w,loaded,variant['rows'],prefix)['status']==variant['expected']
        for recorded in result['pilot_retrospective_diagnostics']:
            path0=ROOT/recorded['source'];assert hashlib.sha256(path0.read_bytes()).hexdigest()==recorded['source_sha256']
            actual=json.loads(path0.read_text(encoding='utf8'));w=by_id[recorded['workflow']]
            loaded,prefix=runner.fresh(w);runner.restore(loaded,actual['post_state'])
            assert check_formal(w,loaded,actual['receivers'],prefix,actual['native_utility'],actual['native_utility_error'])==recorded['score']
    else:p.error('Use --qualify for disposable local controls or --check for saved offline evidence')
    print(json.dumps({'status':'PASS','workflow_count':18,'fixture_tasks':18,'pilot_arms':12,'new_model_calls':0,'formal_online_gate':'closed'}))

if __name__=='__main__':main()

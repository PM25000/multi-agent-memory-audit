"""Conservative post-hoc pilot diagnostic v3; preserves v2/code/raw artifacts.

Later explicit field blocks after a new heading cannot inherit first-block True.
This is a deterministic reply-format diagnostic, not semantic human gold.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from unittest.mock import patch

import same_input_public_bindings_v2 as v2

OUT=v2.BASE/'pilot/offline_grader_amendment_v3'
original_bound_fields=v2.bound_fields

def bound_fields(text,target,candidates,expected_rating,expected_range=None,fixed_named=False):
    result=original_bound_fields(text,target,candidates,expected_rating,expected_range,fixed_named)
    if result['target_binding'] is not True:
        return result
    lines=v2.clean(text).splitlines()
    anchors=[i for i,l in enumerate(lines) if re.match(r'^\s*(?:Hotel Name|Hotel)\s*[:=]',l,re.I)]
    if anchors:
        start=anchors[0]
    elif not fixed_named:
        starts=[i for i,l in enumerate(lines) if re.match(r'^\s*#*\s*(?:Recommended Hotel|Recommendation)\b',l,re.I)]
        start=starts[0] if len(starts)==1 else -1
    else:
        start=0
    headings=[i for i in range(start+1,len(lines)) if re.match(r'^\s*#+\s*\S',lines[i])]
    if headings and any(re.match(r'^\s*(?:Rating|Rated|Price\s+Range)\s*[:=]',l,re.I) for l in lines[headings[0]+1:]):
        return {'target_binding':None,'rating':None,'price_range':None,
          'reason':'later_explicit_fields_after_heading_unresolved'}
    return result

def extra_controls():
    head='Hotel Name: Target Hotel\nRating: 4.7\nPrice range: 110 - 200\n'
    cases=[
      ('late_same_object_correction',head+'## Correction for Target Hotel\nRating: 4.2\nPrice range: 999 - 1000',None,None,None),
      ('later_generic_heading_fields',head+'## Additional details\nRating: 4.7\nPrice range: 110 - 200',None,None,None),
      ('multiple_heading_late_fields',head+'## Notes\nSome notes.\n## Additional details\nRating: 4.7',None,None,None),
      ('late_heading_without_fields',head+'## Notes\nNo further numeric fields.',True,True,True),
      ('rival_before_explicit_target','Rival Hotel Rating: 4.2\n'+head,True,True,True),
    ]
    results=[]
    for key,text,b,r,p in cases:
        got=bound_fields(text,'Target Hotel',['Target Hotel','Rival Hotel'],4.7,(110,200))
        expected=[b,r,p]
        passed=[got['target_binding'],got['rating'],got['price_range']]==expected
        results.append({'name':key,'passed':passed,'actual':got,'expected':expected})
    assert all(x['passed'] for x in results)
    return results

def build():
    # In-process diagnostic substitution only; no imported file or artifact edit.
    with patch.object(v2,'bound_fields',bound_fields):
        result=v2.build()
    counterexample='Hotel Name: Target Hotel\nRating: 4.7\nPrice range: 110 - 200\n## Correction for Target Hotel\nRating: 4.2\nPrice range: 999 - 1000'
    result['controls']+=extra_controls()
    result['version']='same-input-pilot-offline-public-binding-amendment-v3'
    result['v2_retained_failed_boundary']={
      'fixture':counterexample,
      'v2_actual':original_bound_fields(counterexample,'Target Hotel',['Target Hotel','Rival Hotel'],4.7,(110,200)),
      'v3_actual':bound_fields(counterexample,'Target Hotel',['Target Hotel','Rival Hotel'],4.7,(110,200)),
      'interpretation':'v2 first-block acceptance is superseded for later explicit field sections; old v2 artifacts unchanged'}
    for row in result['arms']:
        row['binding_v3']=row.pop('binding_v2')
    for p in (Path(__file__),v2.OUT/'diagnostic.json'):
        result['source_sha256'][str(p.relative_to(v2.ROOT)).replace('\\','/')]=v2.sha(p)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',action='store_true')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    result=build()
    path=OUT/'diagnostic.json'
    if a.build:
        OUT.mkdir(parents=True,exist_ok=True)
        assert not path.exists()
        path.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8')
    elif a.check:
        assert v2.read(path)==result, 'Source/diagnostic drift'
    print(json.dumps({'status':'PASS','arms':12,'controls':len(result['controls']),
      'new_model_calls':0,'posthoc':True,
      'revised_counts':{str(v):sum(x['revised_public_bounded_diagnostic'] is v for x in result['arms']) for v in (True,False,None)}}))

if __name__=='__main__':main()

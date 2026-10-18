"""Post-hoc pilot reply-binding diagnostic, 0 API, no frozen artifact mutation.

Supports only the fixed travel Task0/4 software pilot. Native actions, state,
old scores and model responses remain untouched. No semantic human gold.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/results/same_input_independent_20261005'
OUT = BASE / 'pilot/offline_grader_amendment_v2'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clean(text):
    # Common presentation wrappers only; never drop words or numbers.
    return re.sub(r'^[ \t]*[-*+]\s+', '', text.replace('**', '').replace('__', '').replace('`', ''), flags=re.M)

def bound_fields(text, target, candidates, expected_rating, expected_range=None, fixed_named=False):
    """Explicit name-bound contiguous field block, not first number in reply."""
    lines = clean(text).splitlines()
    anchors = []
    for i, line in enumerate(lines):
        m = re.match(r'^\s*(?:Hotel Name|Hotel)\s*[:=]\s*(.*?)\s*$', line, re.I)
        if m:
            anchors.append((i, m.group(1)))
    if len(anchors) > 1:
        return {'target_binding': None, 'rating': None, 'price_range': None, 'reason': 'multiple_explicit_object_anchors'}
    if anchors:
        i, selected = anchors[0]
        if selected != target:
            return {'target_binding': False, 'rating': None, 'price_range': None, 'reason': 'explicit_other_target'}
        block = []
        for line in lines[i+1:]:
            if re.match(r'^\s*#', line):
                break
            if any(x != target and x in line for x in candidates):
                break
            block.append(line)
    elif fixed_named:
        # Task0 already names the sole requested object in the user request.
        if any(x != target and x in text for x in candidates):
            return {'target_binding': None, 'rating': None, 'price_range': None, 'reason': 'rival_in_fixed_target_reply'}
        block = lines
    else:
        # Bounded supported alternative: recommendation heading immediately names
        # the target; comparisons before that heading are outside the block.
        starts = [i for i,l in enumerate(lines) if re.match(r'^\s*#*\s*(?:Recommended Hotel|Recommendation)\b', l, re.I)]
        if len(starts) != 1:
            return {'target_binding': None, 'rating': None, 'price_range': None, 'reason': 'no_unique_explicit_target_block'}
        block = lines[starts[0]+1:]
        names = [x for x in candidates if any(x in l for l in [lines[starts[0]]] + block)]
        if names != [target]:
            return {'target_binding': False if len(names)==1 else None, 'rating': None, 'price_range': None, 'reason': 'recommendation_target_unclear_or_wrong'}
    ratings = []
    ranges = []
    for line in block:
        m = re.fullmatch(r'\s*(?:Rating|Rated)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*5)?\s*', line, re.I)
        if m:
            ratings.append(float(m.group(1)))
        m = re.fullmatch(r'\s*Price\s+Range\s*[:=]\s*[$€£]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*[$€£]?\s*([0-9]+(?:\.[0-9]+)?)\s*', line, re.I)
        if m:
            ranges.append(tuple(float(x) for x in m.groups()))
    return {'target_binding': True,
            'rating': None if not ratings else all(x==expected_rating for x in ratings),
            'price_range': None if expected_range is None or not ranges else all(x==tuple(expected_range) for x in ranges),
            'reason': 'explicit_source_target_block' if anchors else 'fixed_named_user_target' if fixed_named else 'explicit_recommendation_block'}

def controls():
    target, rival = 'Target Hotel', 'Rival Hotel'
    names = [target, rival]
    cases = [
      ('markdown_correct', '**Hotel Name:** Target Hotel\n**Rating:** 4.7\n**Price Range:** 110 - 200', True, True, True),
      ('plain_correct', 'Hotel Name: Target Hotel\nRating: 4.7\nPrice range: 110 - 200', True, True, True),
      ('rival_correct_target_wrong', 'Rival Hotel: Rating: 4.7\nHotel Name: Target Hotel\nRating: 4.2\nPrice range: 110 - 200', True, False, True),
      ('rival_conflict_target_correct', 'Rival Hotel: Rating: 4.2\nPrice range: 999 - 1000\nHotel Name: Target Hotel\nRating: 4.7\nPrice range: 110 - 200', True, True, True),
      ('reverse_range', 'Hotel Name: Target Hotel\nRating: 4.7\nPrice range: 200 - 110', True, True, False),
      ('wrong_target_same_values', 'Hotel Name: Rival Hotel\nRating: 4.7\nPrice range: 110 - 200', False, None, None),
      ('two_object_blocks_unknown', 'Hotel Name: Rival Hotel\nRating: 4.2\nHotel Name: Target Hotel\nRating: 4.7\nPrice range: 110 - 200', None, None, None),
      ('missing_label_unknown', 'Hotel Name: Target Hotel\n4.7 and 110 and 200', True, None, None),
      ('conflicting_target_values', 'Hotel Name: Target Hotel\nRating: 4.7\nRating: 4.2\nPrice range: 110 - 200', True, False, True),
      ('recommendation_section', 'Rival Hotel rating 4.2\n## Recommended Hotel: Target Hotel\nRating: 4.7\nPrice range: 110 - 200', True, True, True),
      ('fixed_named_markdown', '**Rating:** 4.7\n**Price range:** 110 - 200', True, True, True),
      ('rival_range_correct_target_reverse', 'Rival Hotel: Price range: 110 - 200\nHotel Name: Target Hotel\nRating: 4.7\nPrice range: 200 - 110', True, True, False),
      ('wrong_explicit_anchor_then_right_recommendation', 'Hotel Name: Rival Hotel\nRating: 4.7\nPrice range: 110 - 200\n## Recommended Hotel: Target Hotel\nRating: 4.7\nPrice range: 110 - 200', False, None, None),
    ]
    result=[]
    for key,text,b,r,p in cases:
        got=bound_fields(text,target,names,4.7,(110,200),fixed_named=key=='fixed_named_markdown')
        passed=(got['target_binding'],got['rating'],got['price_range'])==(b,r,p)
        result.append({'name':key,'passed':passed,'actual':got,'expected':[b,r,p]})
    assert all(x['passed'] for x in result)
    return result

def build():
    data=read(BASE/'dataset.json')
    workflows={w['id']:w for w in data['workflows']}
    rows=[]
    source_paths={Path(__file__),BASE/'dataset.json',BASE/'freeze.json',
      ROOT/'experiments/analyze_same_input_pilot.py',ROOT/'experiments/same_input_travel_adapter.py',
      ROOT/'experiments/run_same_input_independent.py', ROOT/'docs/paper/same-input-independent-protocol-20261005.md'}
    for model in ('glm','qwen'):
        scope=BASE/'pilot'/model
        excluded=(scope/'api/scope_halted.json').exists()
        source_paths.update((scope/'api/responses').glob('*.json'))
        if excluded:
            source_paths.add(scope/'api/scope_halted.json')
        for task in (0,4):
            wid=f'travel/user_task_{task}'
            w=workflows[wid]
            source_paths.add(scope/'repeat0'/wid.replace('/','__')/'selection.json')
            # Expected values originate only in already frozen legal native reads.
            maps={}
            for r in w['records']:
                kind=r['source']['record_kind']
                if isinstance(r['original_value'],dict): maps.setdefault(kind,{}).update(r['original_value'])
            if task==0:
                target='Le Marais Boutique'
            else:
                eligible=[n for n,v in maps['price'].items() if float(v.rsplit(' - ',1)[1])<210]
                target=max(eligible,key=lambda n:float(maps['rating_reviews'][n].splitlines()[0].split(': ')[1]))
            value=float(maps['rating_reviews'][target].splitlines()[0].split(': ')[1])
            bounds=None if task==0 else tuple(float(x) for x in maps['price'][target].split(': ')[1].split(' - '))
            for arm in ('D','U','F'):
                p=scope/'repeat0'/wid.replace('/','__')/(arm+'.json')
                source_paths.add(p)
                original=read(p)
                reports=[r for r in original['receivers'] if r['id']=='r0']
                assert len(reports)==1
                binding=bound_fields('\n'.join(reports[0]['text']),target,list(maps['rating_reviews']),value,bounds,task==0)
                old=original['business_check']
                checks=deepcopy(old['public_checks'])
                checks['report_explicit_source_target_binding_v2']=binding['target_binding']
                checks['report_labelled_rating']=binding['rating']
                if task==4: checks['report_price_range']=binding['price_range']
                failed=[k for k,v in checks.items() if v is False]
                unknown=[k for k,v in checks.items() if v is None]
                bounded=False if failed else None if unknown else True
                rows.append({'model':model,'workflow':wid,'arm':arm,'excluded_model_scope':excluded,
                  'source':str(p.relative_to(ROOT)).replace('\\','/'),'old_score_unchanged':old,
                  'binding_v2':binding,'revised_public_bounded_diagnostic':None if excluded else bounded,
                  'revised_failed_checks':failed,'revised_unknown_checks':unknown,
                  'native_utility_unchanged':original['native_utility'],
                  'scope':'Post-hoc deterministic reply-format diagnostic; no human semantics gold, no action/state/native score modification'})
    return {'version':'same-input-pilot-offline-public-binding-amendment-v2','posthoc':True,
      'new_model_calls':0,'human_validated':False,'formal_efficacy_result':False,
      'controls':controls(),'arms':rows,
      'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(source_paths)}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    result=build()
    path=OUT/'diagnostic.json'
    if args.build:
        OUT.mkdir(parents=True,exist_ok=True)
        assert not path.exists(), 'Do not overwrite an earlier diagnostic'
        path.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    elif args.check:
        assert read(path)==result, 'Diagnostic or source drift'
    print(json.dumps({'status':'PASS','arms':len(result['arms']),'controls':len(result['controls']),
      'new_model_calls':0,'posthoc':True,'revised_counts':{str(v):sum(x['revised_public_bounded_diagnostic'] is v for x in result['arms']) for v in (True,False,None)}}))

if __name__=='__main__': main()

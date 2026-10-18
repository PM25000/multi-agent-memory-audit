"""Standard-library, read-only semantic-provenance/count replay. Never calls models."""
from pathlib import Path
from collections import Counter
import hashlib,itertools,json,math,random,runpy,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
read=lambda n:json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def gid(key,e):return 'G'+hashlib.sha256(canon(['broadcast-group-review-1',key,e])).hexdigest()[:12].upper()
def outcome(group,vector,model,endpoint):
 opts=[]
 for t in group['active_tasks']:
  status=vector['needs'][f'T{t+1}'];status=status.get('status') if isinstance(status,dict) else status
  if vector['fact_status']!='supported' or status in ('','uncertain'):opts.append((0,1))
  else:opts.append((int(status=='required' or endpoint=='lower' and status=='possible_only'),))
 pairs=[]
 for bits in itertools.product(*opts):
  needed={t for t,b in zip(group['active_tasks'],bits) if b}
  y=len(set(group['added'][model])-needed)
  pairs.append((y,y if needed&set(group['selectors'][model]) else 0))
 return [[min(x[j] for x in pairs),max(x[j] for x in pairs)] for j in (0,1)]
def verify():
 manifest=read('manifest.json')
 for filename,expected in manifest['files_sha256'].items():
  assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected,filename
 cases=read('evidence/cases.json');assert len(cases)==150
 original=read('construction/benchmark_150.json');composites=read('construction/composites.json');atoms=read('construction/atomic-subtasks.json');card=read('evidence/benchmark-data-card.json')
 assert [c['item'] for c in cases]==original
 cm={c['id']:c for c in composites};at={a['name']:a for a in atoms}
 assert {x['composite_id'] for x in original}==set(cm)
 assert all(set(x['ground_truth']['subtasks'])==set(cm[x['composite_id']]['subtask_names']) for x in original)
 assert {str(k):v for k,v in Counter(len(x['ground_truth']['subtasks']) for x in original).items()}==card['nominal_task_count_distribution']
 assert {str(k):v for k,v in Counter(len(c['active_tasks']) for c in cases).items()}==card['active_task_count_distribution']
 assert dict(Counter(at[n]['scenario'] for x in original for n in x['ground_truth']['subtasks']))==card['nominal_task_scenario_memberships']
 assert sum(len({at[n]['scenario'] for n in x['ground_truth']['subtasks']})>1 for x in original)==149
 assert len({n for x in original for n in x['ground_truth']['subtasks']})==63
 sels=read('evidence/selections.json');grades=read('evidence/retained-grades.json');models=list(sels)
 assert set(models)=={'qwen','glm','gpt55_responses'}
 for c in cases:
  memory=[e['id'] for e in c['entities'] if e['source']=='memory']
  for e in c['entities']:
   text=c['item']['memory_context' if e['source']=='memory' else 'user_input']
   assert e['quote'] in text and e['value'] in e['quote'],(c['key'],e['id'])
  for m in models:
   rec=grades[m][c['key']];assert rec['memory_entity_ids']==memory
   assert {g['task_index'] for g in rec['tasks']}==set(c['active_tasks'])
   selected={str(g['task_index']):sorted(e for e,v in zip(memory,g['grades']) if v==3) for g in rec['tasks']}
   assert selected==sels[m][c['key']],(m,c['key'])
 frame=[];relations=set()
 for c in sorted(cases,key=lambda c:c['key']):
  active=sorted(c['active_tasks'])
  for e in sorted(x['id'] for x in c['entities'] if x['source']=='memory'):
   selectors={m:[t for t in active if e in sels[m][c['key']][str(t)]] for m in models}
   added={m:[t for t in active if t not in selectors[m]] if selectors[m] else [] for m in models}
   if not any(added.values()):continue
   frame.append({'group_id':gid(c['key'],e),'case_key':c['key'],'entity_id':e,'active_tasks':active,'task_count':len(active),'selectors':selectors,'added':added})
   for ts in added.values():relations.update((c['key'],e,t) for t in ts)
 assert frame==read('evidence/frame.json')
 assert len(frame)==1644 and len(relations)==2976 and len({g['case_key'] for g in frame})==149
 protocol=read('evidence/protocol.json')
 assert hashlib.sha256(canon(frame)).hexdigest()==protocol['frame_sha256']
 sample=[]
 for h,n in ((2,34),(3,42),(4,20)):
  pool=[g for g in frame if g['task_count']==h];rng=random.Random(20260923+h);picked=rng.sample(pool,n);rng.shuffle(picked)
  for i,g in enumerate(picked):sample.append({**g,'batch':1 if i<n//2 else 2,'inclusion_probability':n/len(pool),'design_weight':len(pool)/n})
 sample.sort(key=lambda g:(g['batch'],g['case_key'],g['entity_id']))
 assert sample==read('evidence/sample.json')
 assert hashlib.sha256(canon(sample)).hexdigest()==protocol['sample_sha256']
 context=read('evidence/reviewer-context.json');bycase={c['key']:c for c in cases}
 public_groups={g['id']:g for g in context['groups']}
 for g in sample:
  pc=context['cases'][public_groups[g['group_id']]['case_id']];c=bycase[g['case_key']]
  assert pc['memory']==c['item']['memory_context'] and pc['request']==c['item']['user_input']
  assert pc['tasks']==[{'id':f'T{t["task_index"]+1}','name':t['name'],'quotes':t['request_quotes']} for t in c['fixed_tasks'] if t['task_index'] in g['active_tasks']]
  assert pc['entities']==[{'code':f'F{i:02}','label':e['label'],'value':e['value'],'source':e['source'],'quote':e['quote']} for i,e in enumerate(c['entities'],1)]
  e=next(e for e in c['entities'] if e['id']==g['entity_id'])
  assert public_groups[g['group_id']]['value']==e['value'] and public_groups[g['group_id']]['quote']==e['quote']
 sheets={r:read(f'evidence/judgments_{r}.json') for r in ('A','B','C')}
 assert all(sheets[r]['declared_independent'] and sheets[r]['prior_participation']=='no' for r in ('A','B'))
 assert sheets['C']['declared_adjudicated']
 assert all(set(sheets[r]['answers'])=={g['group_id'] for g in sample} for r in ('A','B'))
 assert set(sheets['C']['answers'])<={g['group_id'] for g in sample}
 resolved={};counts=Counter()
 for g in sample:
  k=g['group_id'];a,b,c=[sheets[r]['answers'].get(k,{}) for r in ('A','B','C')]
  out={'fact_status':'','needs':{},'provenance':{}}
  for key in ['fact_status']+[f'T{t+1}' for t in g['active_tasks']]:
   values=[x.get('fact_status','') if key=='fact_status' else x.get('needs',{}).get(key,{}).get('status','') for x in (a,b,c)]
   va,vb,vc=values;agree=bool(va and va==vb and va!='uncertain')
   counts[('fact' if key=='fact_status' else 'need')+('_agreement' if agree else '_disagreement_or_missing_uncertain')]+=1
   v=vc or (va if agree else '');assert v and v!='uncertain'
   assert not(vc and agree and vc!=va),'agreement changed'
   out['provenance'][key]='adjudicated' if vc else 'pair_agreement' if agree else 'unresolved'
   if key=='fact_status':out[key]=v
   else:out['needs'][key]=v
  resolved[k]=out
 assert resolved==read('evidence/resolved.json') and dict(counts)==read('evidence/expected-agreement.json')
 assert counts['fact_agreement']==96 and counts['need_agreement']==244 and counts['need_disagreement_or_missing_uncertain']==30
 label_counts=Counter(v for row in resolved.values() for v in row['needs'].values())
 assert label_counts=={'required':101,'possible_only':3,'not_required':170}
 scored={g['group']:g for g in read('numerical-S2/groups.json')['groups']}
 for g in sample:
  for m in models:
   for ep in ('lower','upper'):
    actual=outcome(g,resolved[g['group_id']],m,ep)
    assert all(a==b for a,b in actual)
    assert [p[0] for p in actual]==scored[g['group_id']]['counts'][m][ep]
 sensitivity=read('numerical-S2/sensitivity/contributions.json');sample_by_id={g['group_id']:g for g in sample}
 for row in sensitivity:
  g=sample_by_id[row['group']];k=g['group_id'];m=row['model'];ep=row['endpoint']
  assert math.isclose(row['weight'],g['design_weight'])
  for r in ('A','B'):assert outcome(g,sheets[r]['answers'][k],m,ep)==row['counts'][r]
  assert outcome(g,resolved[k],m,ep)==row['counts']['adjudicated']
 saved_path=list(sys.path);sys.path.insert(0,str(ROOT/'numerical-S2'))
 try:numerical=runpy.run_path(str(ROOT/'numerical-S2/verify.py'))['verify']()
 finally:sys.path[:]=saved_path
 return {'status':'PASS','frame_groups':1644,'sample_groups':96,'sample_inputs':75,'necessity_judgments':274,'reconstructed_frame_and_seeded_sample':True,'reviewer_original_context_matched':True,'grade3_selections_reparsed':450,'original_AB_and_C_to_scored_counts':True,'unchanged_pair_agreements':244,'adjudicated_necessity_disagreements':30,'label_counts':dict(label_counts),'AB_sensitivity_vectors_matched':True,'numerical':numerical,'boundary':'Offline source-to-judgment-to-count inspectability, not automated semantic truth, new annotation or provider rerun.'}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
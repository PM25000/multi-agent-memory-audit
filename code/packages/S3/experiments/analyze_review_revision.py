"""离线重算：固定全集一致性、成对统计、来源覆盖与实际调用账单。"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from review_revision import OUT, EXP, ROOT, SEED, read, dump, cases_all, consensus_reference, apply_source_audit


def agreement(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('agreement_requires_complete_paired_universe')
    ca, cb = Counter(a), Counter(b)
    n = len(a)
    po = sum(x==y for x,y in zip(a,b))/n
    pe = sum(ca[k]*cb[k] for k in set(ca)|set(cb))/n**2
    return {'n':n, 'agreement':po, 'kappa':(po-pe)/(1-pe) if pe<1 else None,
            'confusion':{f'{x}|{y}':v for (x,y),v in Counter(zip(a,b)).items()},
            'marginals_a':dict(ca),'marginals_b':dict(cb)}


def paired(a,b,clusters=None):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if len(a)!=len(b) or not len(a):
        raise ValueError('paired_universe_mismatch')
    d=b-a
    rng=np.random.default_rng(SEED)
    if clusters is None:
        bs=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
    else:
        # Resample original items, keeping all sampled tool instances together.
        keys=sorted(set(clusters))
        groups=[d[np.asarray(clusters)==k] for k in keys]
        bs=np.asarray([np.concatenate([groups[j] for j in rng.integers(0,len(keys),len(keys))]).mean()
                       for _ in range(10000)])
    return {'n':len(d), 'mean_difference_b_minus_a':float(d.mean()),
            'ci95':np.quantile(bs,[.025,.975]).tolist(),
            'b_higher':int((d>0).sum()),'equal':int((d==0).sum()),'a_higher':int((d<0).sum())}


def aggregate(metrics):
    if not metrics:return None
    n=len(metrics)
    p=[x['P_available'] for x in metrics if x['P_available'] is not None]
    return {'n':n, 'P_available_macro':float(np.mean(p)) if p else None,
            'n_available':len(p),
            'P_all_macro':float(np.mean([x['P_all'] for x in metrics])),
            'P_all_micro':sum(x['correct'] for x in metrics)/sum(x['total'] for x in metrics),
            'P_available_micro':sum(x['correct'] for x in metrics)/sum(x['eligible_total'] for x in metrics)
                              if sum(x['eligible_total'] for x in metrics) else None,
            'eligible_fields':sum(x['eligible_total'] for x in metrics),
            'all_fields':sum(x['total'] for x in metrics),
            **{k:float(np.mean([x[k] for x in metrics])) for k in
               ['L_lower','L_upper','L_all_lower','L_all_upper','memory_transmissions','unique_L_lower']}}


def legacy_kappa():
    d=read(EXP/'results/kappa_validation_v3_thinking.json')
    a,b={},{}
    conflicts=[]
    for target,key in [(a,'per_item_ds'),(b,'per_item_glm')]:
        for item,ev in d[key].items():
            for task,labels in ev.get('detail',{}).items():
                for label in ['correct','missing','leaked']:
                    for field in labels.get(label,[]):
                        k=(item,task,field)
                        if k in target and target[k]!=label:
                            conflicts.append({'evaluator':key,'key':k,'labels':[target[k],label]})
                            # Old evaluator used correct > missing > leaked precedence.
                            continue
                        target[k]=label
    both=sorted(a.keys() & b.keys())
    union=sorted(a.keys() | b.keys())
    return {'intersection':agreement([a[k] for k in both],[b[k] for k in both]),
            'union_with_not_emitted':agreement([a.get(k,'not_emitted') for k in union],
                                               [b.get(k,'not_emitted') for k in union]),
            'one_sided':len(set(union)-set(both)),
            'one_sided_leaked':sum(a.get(k)=='leaked' or b.get(k)=='leaked' for k in set(union)-set(both)),
            'conflicting_legacy_labels':conflicts,
            'interpretation':'not_emitted is missing detection output, not proven absence; field aliases are not harmonized'}


def annotation_summary(annotations):
    need_a,need_b,status_a,status_b,active_a,active_b=[],[],[],[],[],[]
    status=Counter()
    quality=[]
    for a in annotations:
        facts=a['entities']
        ra,rb=a['reference_ds'],a['reference_glm']
        aa={(r['task_index'],r['field']):r for r in ra['bindings']}
        bb={(r['task_index'],r['field']):r for r in rb['bindings']}
        if aa.keys()!=bb.keys():raise ValueError('reference_universe_mismatch')
        for k in sorted(aa):
            status_a.append(aa[k]['status']);status_b.append(bb[k]['status'])
        na,nb={},{}
        for target,rows in [(na,ra['bindings']),(nb,rb['bindings'])]:
            for r in rows:target.setdefault(r['task_index'],set()).update(r['entity_ids'])
        for i in sorted(na):
            for e in facts:
                need_a.append(int(e['id'] in na[i]));need_b.append(int(e['id'] in nb[i]))
        ta={r['task_index']:r['active'] for r in ra['tasks']}
        tb={r['task_index']:r['active'] for r in rb['tasks']}
        for i in sorted(ta):active_a.append(ta[i]);active_b.append(tb[i])
        ref=a['reference']
        status.update(r['status'] for r in ref['bindings'])
        quality.append({'key':a['key'],'facts':len(facts),
                        'active_all':all(r['active'] for r in ref['tasks']),
                        'binding_agreement':sum(r['agreement'] for r in ref['bindings'])/len(ref['bindings']),
                        'unresolved_fields':[{'task_index':r['task_index'],'field':r['field'],'status':r['status']}
                                             for r in ref['bindings'] if r['status']!='available']})
    return {'n':len(annotations),'binding_status':dict(status),
            'necessity_full_universe':agreement(need_a,need_b) if need_a else None,
            'source_availability':agreement(status_a,status_b) if status_a else None,
            'task_activation':agreement(active_a,active_b) if active_a else None,
            'quality':quality}


def cost_summary():
    rows=[];seen=set()
    for p in OUT.glob('*/calls/*.json'):
        r=read(p);uid=r.get('response_id')
        if uid and uid in seen:continue
        if uid:seen.add(uid)
        u=r.get('usage',{})
        rows.append({'owner':p.parent.parent.name,'model':r['request']['model'],
                     'tag':r['tag'],'cost':u.get('cost'),'elapsed_seconds':r['elapsed_seconds'],
                     'prompt_tokens':u.get('prompt_tokens',0),'completion_tokens':u.get('completion_tokens',0)})
    groups={}
    for owner in sorted({r['owner'] for r in rows}):
        rs=[r for r in rows if r['owner']==owner]
        groups[owner]={'responses':len(rs),'reported_usd':sum(r['cost'] or 0 for r in rs),
                       'unpriced_responses':sum(r['cost'] is None for r in rs)}
    return {'groups':groups,'reported_usd':sum(r['cost'] or 0 for r in rows),
            'unpriced_responses':sum(r['cost'] is None for r in rows),'calls':rows,
            'note':'Provider-reported response costs; failed requests with no usage may not be represented. No live FX conversion.'}


def cached_summary():
    summary={}
    all_rows=[]
    for model in ['qwen','glm']:
        records=[read(p) for p in sorted((OUT/'cached_evaluation'/model).glob('*.json'))
                 if not p.stem.endswith('_glm')]
        all_rows.extend(records)
        summary[model]={'complete':len(records),'splits':{}}
        for split in ['multi','single','neutral']:
            by={(r['key'],r['condition']):r for r in records if r['split']==split}
            keys=sorted({k for k,c in by if (k,'ours') in by and (k,'um') in by})
            if not keys:continue
            conditions={c:aggregate([by[k,c]['metrics'] for k in keys]) for c in ['ours','um']}
            differences={}
            for metric in ['P_available','P_all','L_lower','L_upper']:
                ks=[k for k in keys if all(by[k,c]['metrics'][metric] is not None for c in ['ours','um'])]
                differences[metric]=paired([by[k,'ours']['metrics'][metric] for k in ks],
                                            [by[k,'um']['metrics'][metric] for k in ks])
            summary[model]['splits'][split]={'paired_cases':len(keys),'conditions':conditions,
                                            'paired_um_minus_ours':differences}
    aa,bb=[],[];checked=0
    for p in sorted((OUT/'cached_evaluation/qwen').glob('*_glm.json')):
        main=p.with_name(p.name.removesuffix('_glm.json')+'.json')
        if not main.exists():continue
        a,b=read(main),read(p)
        if a['annotation_hash']!=b['annotation_hash'] or a['raw_hash']!=b['raw_hash']:
            raise ValueError('verifier_source_mismatch')
        def labels(rows):
            out={}
            for r in rows:
                for eid in r['present']:out[r['block_index'],eid]=1
                for eid in r['absent']:out[r['block_index'],eid]=0
            return out
        a,b=labels(a['presence']),labels(b['presence'])
        if a.keys()!=b.keys():raise ValueError('verifier_universe_mismatch')
        aa.extend(a[k] for k in sorted(a));bb.extend(b[k] for k in sorted(a));checked+=1
    ag=agreement(aa,bb) if aa else None
    if ag:
        both=sum(a==b==1 for a,b in zip(aa,bb))
        ag['positive_agreement']=2*both/(sum(aa)+sum(bb)) if sum(aa)+sum(bb) else None
        ag['case_conditions']=checked
    summary['presence_verification']=ag
    return summary


def downstream_summary(items):
    from review_downstream import grade_args, CONDITIONS
    from review_value_matching import MATCH_VERSION, MATCH_DESCRIPTION
    rows=[]
    for p in sorted((OUT/'downstream').glob('*.json')):
        r=read(p)
        a=apply_source_audit(read(OUT/'annotations'/f"{r['case_key']}.json"),items[r['case_key']])
        r['metrics']=grade_args(r['parsed'],r['tool'],r['required'],a,r['task_index'])
        rows.append(r)
    by={(r['key'],r['condition']):r for r in rows}
    keys=sorted({k for k,c in by if all((k,t) in by for t in CONDITIONS)})
    stats={}
    for c in CONDITIONS:
        rs=[by[k,c]['metrics'] for k in keys]
        stats[c]={'n':len(rs),'structural_ok':sum(r['structural_ok'] for r in rs),
                  'schema_ok':sum(r['schema_ok'] for r in rs),
                  'fully_referenceable':sum(r['fully_referenceable'] for r in rs),
                  'verified_success':sum(r['verified_success'] for r in rs),
                  'correct':sum(r['correct'] for r in rs),
                  'eligible_total':sum(r['eligible_total'] for r in rs),
                  'total':sum(r['total'] for r in rs)}
    clusters=[by[k,'full_input']['case_key'] for k in keys]
    differences={}
    for c in CONDITIONS[1:]:
        differences[c]=paired([int(by[k,'full_input']['metrics']['verified_success']) for k in keys],
                              [int(by[k,c]['metrics']['verified_success']) for k in keys],clusters) if keys else None
    fixed=paired([int(by[k,'fixed_split']['metrics']['verified_success']) for k in keys],
                  [int(by[k,'fixed_broadcast']['metrics']['verified_success']) for k in keys],clusters) if keys else None
    dump(OUT/'downstream_regraded.json',rows)
    return {'complete_instances':len(keys),'original_items':len(set(clusters)),'conditions':stats,
            'paired_vs_full_input':differences,'broadcast_minus_split':fixed,
            'grading_version':MATCH_VERSION,'grading':MATCH_DESCRIPTION}


def efficiency_summary(records,calls):
    stages={}
    for stage,tag in [('joint',':joint'),('independent_pred',':ind_pred_'),('independent_gold',':ind_gold_')]:
        rs=[c for c in calls if tag in c['tag']]
        stages[stage]={'responses':len(rs),'reported_usd':sum(r['cost'] or 0 for r in rs),
                      'prompt_tokens':sum(r['prompt_tokens'] for r in rs),
                      'completion_tokens':sum(r['completion_tokens'] for r in rs),
                      'response_seconds_median':float(np.median([r['elapsed_seconds'] for r in rs])) if rs else None}
    stages['nominal_calls_per_item']={'joint':1,
          'independent_pred_including_shared_joint':float(np.mean([1+len(r['tasks']) for r in records])) if records else None,
          'independent_gold':float(np.mean([r['m'] for r in records])) if records else None}
    stages['scope']='Nominal calls exclude common annotation, mapping evaluation and schema repairs. Response times include remote/network and possible retry time; not end-to-end local deployment latency.'
    return stages


def summarize():
    items={c['key']:c['item'] for c in cases_all()}
    annotations=[apply_source_audit(read(p),items[p.stem]) for p in sorted((OUT/'annotations').glob('*.json'))]
    an=annotation_summary(annotations)
    quality={r['key']:r for r in an['quality']}
    summary={'protocol':'source-reference-consensus-1','annotations':an,'legacy_kappa':legacy_kappa(),
             'models':{},'cost':cost_summary(),'cached_evaluation':cached_summary(),
             'downstream':downstream_summary(items)}
    for model in ['qwen','glm','gpt55_responses']:
        records=[read(p) for p in sorted((OUT/model/'cases').glob('*.json'))]
        records=[r for r in records if r.get('complete')]
        result={}
        for split in ['multi','single','neutral']:
            rs=[r for r in records if r['split']==split]
            if not rs:continue
            result[split]={}
            for population,subset in [('all',rs),('active_tasks',[r for r in rs if quality[r['key']]['active_all']])]:
                if not subset:continue
                conds=subset[0]['metrics'].keys()
                stats={c:aggregate([r['metrics'][c] for r in subset]) for c in conds}
                comparisons={}
                for target in ['fixed_broadcast','independent_pred','independent_gold']:
                    comp={}
                    for metric in ['P_all','P_available','L_lower','L_upper','memory_transmissions']:
                        paired_rs=[r for r in subset if r['metrics']['fixed_split'][metric] is not None
                                   and r['metrics'][target][metric] is not None]
                        comp[metric]=paired([r['metrics']['fixed_split'][metric] for r in paired_rs],
                                            [r['metrics'][target][metric] for r in paired_rs])
                    comparisons[target]=comp
                result[split][population]={'n':len(subset),'conditions':stats,'paired_vs_fixed_split':comparisons}
        summary['models'][model]={'complete':len(records),'splits':result,
            'efficiency':efficiency_summary(records,[r for r in summary['cost']['calls'] if r['owner']==model]),
            'one_block_control':{'n':sum(len(r['tasks'])==1 for r in records),
               'all_identical':all(r['outputs']['fixed_split']==r['outputs']['fixed_broadcast'] for r in records if len(r['tasks'])==1)},
            'source_repairs_cases':sum(any(t.get('request_source_repair') for t in r['tasks']+r['independent_gold_tasks']) for r in records),
            'missing_payloads':[]}
    dump(OUT/'summary.json',summary)
    return summary


if __name__=='__main__':
    s=summarize()
    print(json.dumps({'annotations':s['annotations']['n'],
                       'completed':{m:r['complete'] for m,r in s['models'].items()},
                       'cost_usd':s['cost']['reported_usd'], 'legacy_kappa':s['legacy_kappa']},ensure_ascii=False,indent=2))

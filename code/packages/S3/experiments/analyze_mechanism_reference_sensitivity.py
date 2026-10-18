"""仅改变参考解释的机制复核；固定缓存、required-only 选择及分发。

独立保存结果，不调用模型、不重写原参考或主表。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
import socket
from unittest.mock import patch

from analyze_error_sources_shared_fields import ROOT, OUT, MODELS, METHODS, CATEGORIES, analyze_view, bin_overlap, csv_text
from analyze_llm_reference_sensitivity import rescore, revise_reference, metrics_equal

DEST = OUT/'mechanism_reference_sensitivity_20260912'
REF = OUT/'llm_reference_sensitivity_20260912'
OLD = OUT/'error_sources_shared_fields_20260912'
VARIANTS = ('original','llm_policy','interpretation_envelope')
COHORTS = ('required_only_original','required_only_retried','original_structure')
FIELD_KEYS = ('eligible_fields','split_correct','broadcast_correct','recovered_fields','single_missing_losses',
              'unique_missing_task_fact_pairs','missing_pairs_affecting_multiple_fields')


def required_frame(case, metric):
    memory={e['id'] for e in case['entities'] if e['source']=='memory'}
    if metric['method_output_failure']:
        assert not metric['visibility']
        return [],[],metric['visibility'],[],metric['method_status']
    byindex={t['task_index']:t for t in case['fixed_tasks']}
    tasks=[];mapping=[]
    for block in metric['visibility']:
        ti=block['task_index'];fixed=byindex[ti]
        mapping.append(ti)
        tasks.append({'name':fixed['name'],'request_quotes':fixed['request_quotes'],
                      'keep_entity_ids':sorted(set(block['visible']) & memory)})
    assert mapping==[t['task_index'] for t in case['fixed_tasks']]
    union=set().union(*(set(t['keep_entity_ids']) for t in tasks))
    broadcast=[{**v,'visible':sorted(set(v['visible'])|union)} for v in metric['visibility']]
    return tasks,mapping,metric['visibility'],broadcast,metric['method_status']


def measure(case, frame, reference):
    tasks,mapping,dv,uv,status=frame
    split=rescore(dv,case['entities'],reference)
    broad=rescore(uv,case['entities'],reference)
    return analyze_view({**case,'reference':reference},tasks,mapping,split,broad,status)


def candidate_references(reference, updates):
    """逐字段枚举已冻结的候选集合；不以包络端点代替非单调分量的范围。"""
    for choices in itertools.product(*(u['candidate_necessary_sets'] for u in updates)):
        chosen=[{**u,'entity_ids':list(ids),'verdict':'available' if ids else 'missing'}
                for u,ids in zip(updates,choices)]
        yield [list(ids) for ids in choices],revise_reference(reference,chosen)


def scalar(row):
    f=row['field_analysis']
    out={k:f[k] for k in FIELD_KEYS}
    out['selection_lost_fields']=f['counts'].get('memory_selection_loss',0)
    out['source_unscorable']=f['counts'].get('source_unscorable',0)
    out['fixed_request_endpoint_gap']=f['counts'].get('fixed_request_endpoint_gap',0)
    out['method_output_failure_fields']=f['counts'].get('method_output_failure',0)
    out['single_missing_recovered']=sum(x['broadcast_recovers_field'] and len(x['missing_ids'])==1 for x in f['fields'])
    for ep,e in row['endpoints'].items():
        out.update({f'{ep}.{k}':v for k,v in e['counts'].items()})
        out[f'{ep}.additional_unnecessary']=e['additional_unnecessary']
        out[f'{ep}.additional_transmissions']=e['additional_transmissions']
    return out


def pooled_ratio_bounds(fixed, options, numerator, denominator):
    """有限每输入选择下的聚合比值极值；输入独立选择，非平均单例比值。"""
    n0=sum(x[numerator] for x in fixed);d0=sum(x[denominator] for x in fixed)
    if d0+sum(min(x[denominator] for x in xs) for xs in options)<=0:
        return None
    answers=[]
    for select in (min,max):
        lo,hi=0.,1.
        for _ in range(55):
            r=(lo+hi)/2
            val=n0-r*d0+sum(select(x[numerator]-r*x[denominator] for x in xs) for xs in options)
            if val>0:lo=r
            else:hi=r
        answers.append((lo+hi)/2)
    return answers


def aggregate(rows):
    if not rows:return {'n':0}
    counts=Counter()
    for r in rows:counts.update(scalar(r))
    ratios={ep:counts[f'{ep}.globally_unneeded_spillover']/counts[f'{ep}.additional_unnecessary']
            if counts[f'{ep}.additional_unnecessary'] else None for ep in ('lower','upper')}
    return {'n':len(rows),'failures':sum(r['status']!='ok' for r in rows),'totals':dict(counts),
            'global_spillover_share':ratios,
            'P_split_macro':sum(r['P_split'] for r in rows)/len(rows),
            'reference_strata':{ep:dict(Counter(bin_overlap(r['reference_overlap'][ep]) for r in rows)) for ep in ('lower','upper')},
            'strata_results':{ep:{b:{'n':len(xs),'added_mean':sum(r['endpoints'][ep]['additional_transmissions'] for r in xs)/len(xs),
                                  'recovered_fields':sum(r['field_analysis']['recovered_fields'] for r in xs)}
                                 for b in ('0-.25','.25-.5','.5-.75','.75-1','undefined')
                                 if (xs:=[r for r in rows if bin_overlap(r['reference_overlap'][ep])==b])}
                             for ep in ('lower','upper')}}


def alternative_bounds(rows, alternatives):
    fixed=[scalar(r) for r in rows if r['key'] not in alternatives]
    opts=[[x['scalars'] for x in alternatives[r['key']]] for r in rows if r['key'] in alternatives]
    sums={k:sum(s[k] for s in fixed) for k in scalar(rows[0])}
    return {'n':len(rows),'changed_inputs':len(opts),
            'total_ranges':{k:[v+sum(min(x[k] for x in xs) for xs in opts),v+sum(max(x[k] for x in xs) for xs in opts)] for k,v in sums.items()},
            'global_spillover_share_ranges':{ep:pooled_ratio_bounds(fixed,opts,f'{ep}.globally_unneeded_spillover',f'{ep}.additional_unnecessary') for ep in ('lower','upper')},
            'stratum_count_ranges':{ep:{b:[sum(bin_overlap(r['reference_overlap'][ep])==b for r in rows if r['key'] not in alternatives)+sum(all(x['strata'][ep]==b for x in alternatives[r['key']]) for r in rows if r['key'] in alternatives),
                                              sum(bin_overlap(r['reference_overlap'][ep])==b for r in rows if r['key'] not in alternatives)+sum(any(x['strata'][ep]==b for x in alternatives[r['key']]) for r in rows if r['key'] in alternatives)]
                                        for b in ('0-.25','.25-.5','.5-.75','.75-1','undefined')} for ep in ('lower','upper')}}


def report(result):
    lines=['# 参考修订后的错误来源、共享程度与字段损失', '',
           '复用既有缓存，采用“仅保留模型判断为合同明确需要的事实”（required-only）。模型判断、载荷、请求可见性和分发方式保持固定，只改变参考解释。没有新增模型调用。',
           '17 项已完成 LLM 复核、规则澄清及先前指标影响检查。本轮接入其结果，检验新的机制分解；不会将原始答案差异直接视为实质性错误。', '',
           '## 三批缓存分别报告', '',
           '- required_only_retried：现稿已补重试传输失败的完整缓存，主读结果。',
           '- required_only_original：此前含接口失败的缓存，用于与上轮诊断逐项对齐；共同成功子集仍按原方法状态选择。',
           '- original_structure：原固定判断拆分／广播实验；指令与合同设置不同，不将跨批次变化归因于参考修订。', '',
           '## 多任务联合路径：全局非必要扩散占新增非必要传输的比例', '',
           '每格依次为披露下端点／上端点参考，非统计置信区间。采用各批次全输入；含失败批次另有共同成功统计。', '',
           '| 缓存 | 模型 | 原参考 | 修订规则 | 已存解释包络 |', '|---|---|---:|---:|---:|']
    for co,models in result['cohorts'].items():
        for model,methods in models.items():
            cells=[]
            for v in VARIANTS:
                s=methods['joint'][v]['multi']['all_inputs']
                cells.append(' / '.join(f"{100*s['global_spillover_share'][ep]:.2f}%" for ep in ('lower','upper')))
            lines.append(f'| {co} | {model} | '+' | '.join(cells)+' |')
    lines+=['','## 现稿 required-only：字段损失与补回', '',
            '| 模型/方法 | 参考 | 可评分字段 | 筛选丢失 | 仅缺一条 | 广播补回 | 其中仅缺一条的补回 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for model,methods in result['cohorts']['required_only_retried'].items():
        for method,data in methods.items():
            for v in ('original','llm_policy'):
                t=data[v]['multi']['all_inputs']['totals']
                lines.append(f'| {model}/{method} | {v} | '+' | '.join(str(t[k]) for k in ('eligible_fields','selection_lost_fields','single_missing_losses','recovered_fields','single_missing_recovered'))+' |')
    lines+=['','## 逐字段备选解释的枚举', '',
            '已存 interpretation_envelope 的字段保留评分沿用修订规则；它只对必要集合给包络，不能替代字段损失的备选解释检查。且四类分量、共享程度都不保证在必要集合的并／交端点取得极值。',
            '本轮对已冻结的 10 个字段候选集合按输入枚举，覆盖 7 条输入、36 个输入内组合；未复核字段仍使用原参考规则。非空候选集合表示可用且要求全部组成，空集合表示该字段不进入可评分分母；不把空集合当成自动通过。',
            '各输入可独立取候选解释，计数范围按输入极值求和；全局非必要扩散占比用总分子／总分母的有限组合极值计算，绝不平均单例比例或把分子分母分别极值相除。', '',
            '| 模型/联合 | 可评分字段范围 | 筛选丢失范围 | 仅缺一条范围 | 广播补回范围 | 全局扩散占比范围（下端点／上端点） |',
            '|---|---:|---:|---:|---:|---:|']
    for model,methods in result['alternatives']['required_only_retried'].items():
        b=methods['joint']['multi']['all_inputs'];t=b['total_ranges']
        spans=['–'.join(map(str,t[k])) for k in ('eligible_fields','selection_lost_fields','single_missing_losses','recovered_fields')]
        spans.append(' / '.join('–'.join(f'{100*x:.2f}%' for x in b['global_spillover_share_ranges'][ep]) for ep in ('lower','upper')))
        lines.append(f'| {model} | '+' | '.join(spans)+' |')
    lines+=['','## 参考共享程度是否改变分组', '',
            '| 集合 | 参考 | 端点 | 低共享 [0,.25) | [.25,.5) | [.5,.75) | 高共享 [.75,1] | 未定义 |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    # 参考分布与模型无关，所有现稿输入均成功，取 Qwen 联合仅为读取同一分布。
    for sp in ('multi','neutral'):
        for v in VARIANTS:
            s=result['cohorts']['required_only_retried']['qwen']['joint'][v][sp]['all_inputs']
            for ep in ('lower','upper'):
                c=s['reference_strata'][ep]
                lines.append(f'| {sp} | {v} | {ep} | '+' | '.join(str(c.get(b,0)) for b in ('0-.25','.25-.5','.5-.75','.75-1','undefined'))+' |')
    lines+=['','变化的逐输入记录、字段及分组迁移见 changes.json 和 changes.csv；所有组成分量、全输入与共同成功结果见 summary.json。备选解释下的每组人数范围另列，不把单一包络分组当作所有解释都相同。', '',
            '## 证据边界与复现', '',
            '- 这是一轮已知解释范围内的事后敏感性检验，不是全数据重新标注或独立验证。复核没有改变模型见过的候选；漏项未被补造为模型选择。',
            '- 不把标签变化造成的统计变化称为模型能力变化；不把字段恢复与传输次数直接换算为统一业务价值。',
            '- 原参考逐项复现上轮诊断；三参考评分与已存参考敏感性产物交叉核对。四类分量和字段补回逐输入检查守恒。',
            '- source_manifest.json 记录全部使用的输入及参考哈希，执行前后核对一致；原结果、人工答卷、参考协议及主表未由本分析覆盖。',
            '- 复现：`python experiments/analyze_mechanism_reference_sensitivity.py`；只读验证追加 `--check`。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    sources={}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8'))
    outputs={};changes=[];change_csv=[]
    with patch.object(socket,'socket',side_effect=RuntimeError('offline analysis: network disabled')):
        cases=read(OUT/'fair_comparison_ids/datasets/test.json')
        if isinstance(cases,dict):cases=cases['cases']
        policy=read(REF/'protocol.json');references=read(REF/'references.json')
        updates=defaultdict(list)
        for u in policy['changes']:
            if u['key'].startswith('binding:'):updates[u['case_key']].append(u)
        assert len(updates)==7 and sum(map(len,updates.values()))==10
        variants={}
        for c in cases:
            key=c['key'];assert references[key]['original']==c['reference']
            assert references[key]['llm_policy']==revise_reference(c['reference'],updates[key])
            assert references[key]['interpretation_envelope']==revise_reference(c['reference'],updates[key],True)
            if updates[key]:variants[key]=list(candidate_references(c['reference'],updates[key]))
        assert sum(map(len,variants.values()))==36
        result={'version':'mechanism-reference-sensitivity-1','new_model_calls':0,'selection_rule':'required-only',
                'cohorts':{},'alternatives':{},'candidate_combinations':{k:len(v) for k,v in variants.items()}}
        for co in COHORTS:
            result['cohorts'][co]={};result['alternatives'][co]={}
            for model in MODELS:
                methods=('joint',) if co=='original_structure' else METHODS
                existing=read(OLD/'per_input'/f"{'original_structure' if co=='original_structure' else 'required_only'}_{model}.json")
                if co!='original_structure':
                    folder='contract_required_only_retried_20260912' if co.endswith('retried') else 'contract_required_only_20260912'
                    metrics=read(OUT/folder/'per_input'/f'{model}.json')['metrics']
                    folder2=OUT/'llm_required_reference_retried_20260912' if co.endswith('retried') else REF
                    scored=read(folder2/f'{model}_required_only_rescored.json')
                    read(OUT/folder/'summary.json')
                rows={m:{v:[] for v in VARIANTS} for m in methods};alts={m:{} for m in methods}
                details=[]
                for n,c in enumerate(cases):
                    key=c['key']
                    if co=='original_structure':
                        raw=read(OUT/model/'cases'/f'{key}.json');old=read(OUT/'request_visibility_revision'/model/f'{key}.json')
                    for m in methods:
                        if co=='original_structure':
                            frame=(raw['tasks'],raw['mapping'],old['metrics']['fixed_split']['visibility'],old['metrics']['fixed_broadcast']['visibility'],'ok')
                        else:
                            assert metrics[m][n]['key']==key
                            frame=required_frame(c,metrics[m][n])
                        for v in VARIANTS:
                            r=measure(c,frame,references[key][v]);rows[m][v].append(r)
                            if v=='original' and co!='required_only_retried':
                                assert json.loads(json.dumps(r))==existing[m][n],(co,model,m,key)
                            if co!='original_structure':
                                assert scored[v][n]['key']==key
                                metrics_equal(rescore(frame[2],c['entities'],references[key][v]),scored[v][n]['metrics'][m])
                        a,b=rows[m]['original'][-1],rows[m]['llm_policy'][-1]
                        if a!=b:
                            field_changes=[{'before':fa,'after':fb} for fa,fb in zip(a['field_analysis']['fields'],b['field_analysis']['fields']) if fa!=fb]
                            record={'cohort':co,'model':model,'method':m,'key':key,'split':c['split'],
                                    'before':scalar(a),'after':scalar(b),'overlap_before':a['reference_overlap'],
                                    'overlap_after':b['reference_overlap'],'field_changes':field_changes}
                            changes.append(record)
                            for k,value in scalar(a).items():
                                if scalar(b)[k]!=value:change_csv.append({k2:record[k2] for k2 in ('cohort','model','method','key','split')}|{'metric':k,'before':value,'after':scalar(b)[k]})
                        if key in variants:
                            opts=[]
                            for ids,ref in variants[key]:
                                r=measure(c,frame,ref)
                                opts.append({'candidate_sets':ids,'scalars':scalar(r),
                                             'strata':{ep:bin_overlap(r['reference_overlap'][ep]) for ep in ('lower','upper')}})
                            alts[m][key]=opts
                both={c['key'] for n,c in enumerate(cases) if all(rows[m]['original'][n]['status']=='ok' for m in methods)}
                result['cohorts'][co][model]={};result['alternatives'][co][model]={}
                for m in methods:
                    result['cohorts'][co][model][m]={v:{sp:{pop:aggregate([r for r in rows[m][v] if r['split']==sp and (pop=='all_inputs' or r['key'] in both)]) for pop in ('all_inputs','both_valid')} for sp in ('multi','single','neutral')} for v in VARIANTS}
                    result['alternatives'][co][model][m]={sp:{pop:alternative_bounds([r for r in rows[m]['llm_policy'] if r['split']==sp and (pop=='all_inputs' or r['key'] in both)],alts[m]) for pop in ('all_inputs','both_valid')} for sp in ('multi','single','neutral')}
                outputs[f'per_input/{co}_{model}.json']=rows
                outputs[f'candidates/{co}_{model}.json']=alts
                print(f'{co}/{model}: frozen views, 3 references, 36 candidate combinations checked',flush=True)
        outputs['summary.json']=result;outputs['changes.json']=changes
        outputs['changes.csv']=csv_text(change_csv,['cohort','model','method','key','split','metric','before','after'])
        outputs['report.md']=report(result)
        outputs['source_manifest.json']=sources
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
        for name,value in outputs.items():
            text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
            path=DEST/name
            if args.check:assert path.read_text(encoding='utf-8')==text,name
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
        print(f'{"Checked" if args.check else "Wrote"} {len(outputs)} artifacts; {len(sources)} unchanged source hashes; no model calls')


if __name__=='__main__':main()

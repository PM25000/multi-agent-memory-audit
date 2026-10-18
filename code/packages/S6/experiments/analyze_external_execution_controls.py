"""Complete planned denominator analysis; no necessity labels or live clients."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import hashlib
import run_external_execution_controls as study
from same_input_native_api import read, digest
from analyze_same_input_formal import usage_accounting

def outcome(row):
    return 'not_run' if row is None else row['scores']['public_check']['status']

def build():
    data,freeze=study.frozen();rows=[];models={};pairs=[];hashes={}
    def keep(p):hashes[str(p.relative_to(study.ROOT)).replace('\\','/')]=study.sha(p)
    for p in (study.OUT/'dataset.json',study.OUT/'freeze.json',study.OUT/'prefix_negative_controls.json'):keep(p)
    negative={x['workflow']:x['native_prefix_only'] for x in read(study.OUT/'prefix_negative_controls.json')}
    for model in study.MODELS:
        model_root=study.OUT/model
        excluded=(model_root/'model_exclusion.json').exists() or bool(list(model_root.glob('*/api/scope_halted.json')))
        receipts=[];orphans=[];preflights=[];identities=Counter()
        for w in data['workflows']:
            root=model_root/w['id'].replace('/','__')
            api=root/'api'
            for p in api.glob('responses/*.json'):
                r=read(p);receipts.append(r);identities[str(r.get('returned_model'))]+=1;keep(p)
            for p in api.glob('intents/*.json'):
                keep(p)
                if not (api/'responses'/p.name).exists():orphans.append(str(p))
            for p in api.glob('preflight_rejections/*.json'):preflights.append(read(p));keep(p)
            for rep in range(study.REPEATS):
                folder=root/f'repeat{rep}'
                selection=read(folder/'selection.json') if (folder/'selection.json').exists() else None
                if selection:keep(folder/'selection.json')
                arms={}
                for arm in study.ARMS:
                    p=folder/(arm+'.json');value=read(p) if p.exists() else None
                    if value is not None:keep(p)
                    arms[arm]=value
                    pc=value['scores']['public_check'] if value else {}
                    actual_received=unknown=0
                    if value:
                        for receiver in value['receivers']:
                            ids=value['real_source_ids'][receiver['id']]
                            if receiver['input_delivery_evidence']=='provider_response_received':actual_received+=len(ids)
                            elif receiver['input_delivery_evidence']=='attempted_reception_unknown':unknown+=len(ids)
                    rows.append({'model':model,'workflow':w['id'],'suite':w['suite'],'repeat':rep,'arm':arm,
                      'excluded_model':excluded,'status':outcome(value),
                      'selection_status':selection['response']['status'] if selection else 'not_run',
                      'not_run_reason':None if value else 'model_excluded' if excluded else 'selection_not_observed' if selection is None else 'selection_failed' if selection['selected'] is None else 'execution_not_completed',
                      'bounded_checks_passed':pc.get('bounded_checks_passed'),'semantics_complete':pc.get('semantics_complete'),
                      'full_state_passed':pc.get('full_state_passed'),'native_prefix_only_control':negative[w['id']],
                      'native_with_prefix':value['scores']['native_with_prefix'] if value else None,
                      'native_executor_only':value['scores']['native_executor_only'] if value else None,
                      'planned_static_relations':value['static_relations'] if value else None,
                      'provider_response_received_static_relations':actual_received,
                      'attempted_unknown_static_relations':unknown,
                      'write_calls':sum(len(r['tool_traces']) for r in value['receivers']) if value else None,
                      'unknown_checks':pc.get('unknown_checks',[]),'failed_checks':pc.get('failed_checks',[]),'errors':pc.get('errors',[])})
                if selection and selection['selected'] is not None:
                    selected=selection['selected'];union=set().union(*(set(v) for v in selected.values()))
                    additions=sum(len(union-set(v)) for v in selected.values())
                else:additions=None
                d,u=arms['D'],arms['U']
                pairs.append({'model':model,'workflow':w['id'],'repeat':rep,'excluded_model':excluded,
                  'D':outcome(d),'U':outcome(u),'P':outcome(arms['P']),'A':outcome(arms['A']),'E':outcome(arms['E']),
                  'U_minus_D_planned_static_relations':additions,
                  'D_U_material_identical':d['static_material']==u['static_material'] if d and u else None,
                  'P_U_static_byte_equal':all(study.nbytes(arms['P']['static_material'][r['id']])==study.nbytes(u['static_material'][r['id']]) for r in w['receivers']) if arms['P'] and u else None,
                  'D_U_state_equal':d['post_state']==u['post_state'] if d and u else None})
        costs=[];missing_cost=0
        for r in receipts:
            if r.get('api_attempted'):
                cost=(r.get('usage') or {}).get('cost')
                if isinstance(cost,(int,float)) and not isinstance(cost,bool) and cost>=0:costs.append(cost)
                else:missing_cost+=1
        models[model]={'excluded_all_effects':excluded,'returned_models':dict(identities),
          'usage':usage_accounting(receipts,orphans,preflights),'provider_reported_cost_sum':sum(costs),
          'attempted_receipts_missing_cost':missing_cost,'cost_currency':'not_verified_no_conversion',
          'usage_by_phase_arm':{label:usage_accounting([r for r in receipts if
            (r['phase']=='selector' if label=='selector' else r['phase']=='executor' and '/'+label+'/' in r['logical_id'])],[],[])
            for label in ('selector',)+study.ARMS},
          'arms':{arm:{'planned':36,'observed':sum(r['status']!='not_run' for r in rows if r['model']==model and r['arm']==arm),
            'status_counts':dict(Counter(r['status'] for r in rows if r['model']==model and r['arm']==arm)),
            'native_true':sum(r['native_with_prefix'] is True for r in rows if r['model']==model and r['arm']==arm),
            'full_state_passed':sum(r['full_state_passed'] is True for r in rows if r['model']==model and r['arm']==arm),
            'bounded_checks_passed':sum(r['bounded_checks_passed'] is True for r in rows if r['model']==model and r['arm']==arm),
            'semantics_complete_true':sum(r['semantics_complete'] is True for r in rows if r['model']==model and r['arm']==arm)} for arm in study.ARMS}}
    return {'version':'external-execution-controls-analysis-1','freeze_sha256':digest(freeze),
      'planned_workflow_arms':360,'observed_workflow_arms':sum(r['status']!='not_run' for r in rows),
      'models':models,'rows':rows,'paired_rows':pairs,'source_hashes':hashes,
      'native_prefix_only_true_tasks':[w for w,v in negative.items() if v is True],
      'human_necessity_gold':False,'independent_world_CI_computed':False,'extra_transfers_are_not_labelled_nonnecessary':True}

def markdown(value):
    lines=['# 外部同输入执行对照：实际结果','',
      '本实验另立执行协议，人工审计继续暂停；没有使用必要性金标。全部计划360工作流臂，当前实际留存'+str(value['observed_workflow_arms'])+'臂。重复和共用套件世界不当独立新任务。','',
      '| 模型 | 条件 | pass | fail | unknown | error | 未运行 | 原生True | 状态一致 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,summary in value['models'].items():
        for arm,s in summary['arms'].items():
            c=s['status_counts'];lines.append('| '+model+('（全模型排除）' if summary['excluded_all_effects'] else '')+' | '+arm+' | '+' | '.join(str(c.get(k,0)) for k in ('pass','fail','unknown','error','not_run'))+' | '+str(s['native_true'])+' | '+str(s['full_state_passed'])+' |')
        usage=summary['usage']
        lines+=['',f"{model}：已确认请求 {usage['confirmed_attempts_with_receipt']}，无回执意图 {usage['intent_without_receipt_count']}；实际返回身份 {json.dumps(summary['returned_models'],ensure_ascii=False)}。真实prompt tokens合计 {usage['recorded_tokens']['prompt_tokens']}，其中缺失字段的已尝试回执 {usage['attempted_receipts_missing_token_field']['prompt_tokens']}。提供商cost数值合计 {summary['provider_reported_cost_sum']:.6f}，缺失 {summary['attempted_receipts_missing_cost']} 次，币种未核实不作人民币换算。",'']
    lines+=['D=独立；U=并集；P=原独立内容+整对象无信息等字节填充；A=完整来源；E=空静态来源。P不是精确token匹配。',
      '', '原生前缀空执行已True任务：'+(', '.join(value['native_prefix_only_true_tasks']) or '无')+'。这些任务原生True不能单独证明执行器完成。',
      '', '| 模型 | 任务/重复 | D | U | P | A | E | U新增静态关系 |', '|---|---|---|---|---|---|---|---:|']
    for p in value['paired_rows']:
        lines.append('| '+p['model']+' | '+p['workflow']+'/'+str(p['repeat'])+' | '+' | '.join(p[a] for a in study.ARMS)+' | '+str(p['U_minus_D_planned_static_relations'])+' |')
    lines+=['','pass仅表示冻结的有限公开约束通过，不是人工验收任意语义。unknown/error/未运行不并入成功。自由总结、未指定年份、格式无法可靠绑定等均保留；静态来源统计不含全部动态工具回执及通知。','',
      '本表不作总体等效、显著优越或非必要披露判断。A失败表示完整输入下仍可能受模型/回合/评分约束；E成功表示原请求或动态信息本身可能足够。需逐任务结合P/A/E解释D/U差异。','',
      '详见 [预运行协议](external-execution-controls-protocol-20261005.md)；逐臂字段、实际回执身份、用量和源哈希见 [完整统计](../../experiments/results/external_execution_controls_20261005_v2/analysis.json)。']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    result=build()
    if args.write:
        (study.OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        (study.ROOT/'docs/paper/external-execution-controls-results-20261005.md').write_text(markdown(result),encoding='utf8')
    print(json.dumps({'planned':360,'observed':result['observed_workflow_arms'],'models':{k:v['arms'] for k,v in result['models'].items()}},ensure_ascii=False))

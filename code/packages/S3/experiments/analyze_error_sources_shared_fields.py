"""冻结缓存的错误来源、参考需求共享程度和完整字段损失诊断。

仅保留等级 3；原结构实验单独报告。只读原缓存，禁止网络，不写论文表。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np

from analyze_contract_required_only import required_tasks
from analyze_fair_comparison_ids import FAIR, MODELS, METHODS, dataset, load_records, build_visibility
from review_revision import ROOT, OUT, apply_source_audit
from review_request_visibility import score_visible

DEST = OUT / 'error_sources_shared_fields_20260912'
SEED = 20260912
REPEATS = 10000
CATEGORIES = ('omission_recovery', 'needed_selected_spillover',
              'needed_but_wrong_recipient_spillover', 'globally_unneeded_spillover')
NAMES = {'qwen': 'Qwen', 'glm': 'GLM', 'gpt55_responses': 'GPT-5.5'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reference_sets(ref, memory, name):
    return {t['task_index']: set(ref[name].get(str(t['task_index']), [])) & memory
            for t in ref['tasks'] if t['active']}


def overlap(sets):
    union = set().union(*sets)
    if len(sets) <= 1 or not union:
        return None
    return (sum(map(len, sets)) - len(union)) / ((len(sets)-1)*len(union))


def bin_overlap(rho):
    return ('undefined' if rho is None else '0-.25' if rho < .25 else
            '.25-.5' if rho < .5 else '.5-.75' if rho < .75 else '.75-1')


def partition(selections, mapping, needs):
    """互斥四分解；参考上需要某事实不意味着模型曾把它选给需要者。"""
    union = set().union(*selections)
    reference_union = set().union(*needs.values())
    correctly_selected = set().union(*(d & needs.get(ti, set()) for d, ti in zip(selections, mapping)))
    pairs = {name: [] for name in CATEGORIES}
    for block, selected in enumerate(selections):
        for eid in sorted(union-selected):
            if eid in needs.get(mapping[block], set()):
                category = 'omission_recovery'
            elif eid not in reference_union:
                category = 'globally_unneeded_spillover'
            elif eid in correctly_selected:
                category = 'needed_selected_spillover'
            else:
                category = 'needed_but_wrong_recipient_spillover'
            pairs[category].append([block, eid])
    counts = {k: len(v) for k, v in pairs.items()}
    expected = sum(len(selections)-sum(e in d for d in selections) for e in union)
    assert sum(counts.values()) == expected
    original_leaks = sum(len(d-needs.get(ti, set())) for d, ti in zip(selections, mapping))
    broadcast_leaks = sum(len(union-needs.get(ti, set())) for ti in mapping)
    assert broadcast_leaks-original_leaks == expected-counts['omission_recovery']
    return {'counts': counts, 'pairs': pairs, 'additional_transmissions': expected,
            'split_unnecessary': original_leaks, 'broadcast_unnecessary': broadcast_leaks,
            'additional_unnecessary': expected-counts['omission_recovery']}


def field_analysis(case, tasks, mapping, split_metric, broadcast_metric, failed):
    memory = {e['id'] for e in case['entities'] if e['source'] == 'memory'}
    bindings = {(b['task_index'], b['field']): b for b in case['reference']['bindings']}
    def bytask(metric):
        result = defaultdict(set)
        for r in metric['visibility']:
            result[r['task_index']].update(r['visible'])
        return result
    dv, uv = bytask(split_metric), bytask(broadcast_metric)
    sf = {(f['task_index'], f['field']): f for f in split_metric['fields']}
    uf = {(f['task_index'], f['field']): f for f in broadcast_metric['fields']}
    assert sf.keys() == uf.keys() == bindings.keys()
    union = set().union(*(set(t['keep_entity_ids']) for t in tasks))
    rows, omitted_field_links, rescue = [], defaultdict(list), defaultdict(set)
    for key, b in bindings.items():
        ti, name = key
        required = set(b['entity_ids'])
        missing = required-dv[ti]
        endpoint = dv[ti] | memory if ti in mapping else set()
        if not sf[key]['eligible']:
            category = 'source_unscorable'
        elif failed:
            category = 'method_output_failure'
        elif ti not in mapping:
            category = 'missing_recipient'
        elif not required <= endpoint:
            category = 'fixed_request_endpoint_gap'
        elif not sf[key]['correct']:
            category = 'memory_selection_loss'
        else:
            category = 'retained'
        recovered = not sf[key]['correct'] and uf[key]['correct']
        assert not sf[key]['correct'] or uf[key]['correct']
        if recovered:
            assert missing and missing <= memory & union
            rescue[ti].update(missing)
        if category == 'memory_selection_loss':
            assert missing and missing <= memory
            for eid in missing:
                omitted_field_links[ti, eid].append(name)
        rows.append({'task_index': ti, 'field': name, 'category': category,
                     'eligible': sf[key]['eligible'], 'required_ids': sorted(required),
                     'required_components': len(required), 'memory_components': len(required & memory),
                     'missing_ids': sorted(missing), 'missing_memory_ids': sorted(missing & memory),
                     'missing_selected_elsewhere_ids': sorted(missing & union),
                     'split_correct': sf[key]['correct'], 'broadcast_correct': uf[key]['correct'],
                     'broadcast_recovers_field': recovered})
    marginal = []
    for (ti, eid), names in sorted(omitted_field_links.items()):
        complete = [r['field'] for r in rows if r['category'] == 'memory_selection_loss'
                    and r['task_index'] == ti and r['missing_ids'] == [eid]]
        marginal.append({'task_index': ti, 'entity_id': eid, 'affected_fields': names,
                         'affected_field_count': len(names), 'single_record_restore_fields': complete,
                         'single_record_restore_count': len(complete)})
    assert sum(r['broadcast_recovers_field'] for r in rows) == broadcast_metric['correct']-split_metric['correct']
    # 参考引导的最小记录-任务补回集合，仅完成广播已补回的固定字段；不是可部署方法。
    for r in rows:
        if r['broadcast_recovers_field']:
            assert set(r['required_ids']) <= dv[r['task_index']] | rescue[r['task_index']]
    counts = Counter(r['category'] for r in rows)
    losses = [r for r in rows if r['category'] == 'memory_selection_loss']
    return {'counts': dict(counts), 'fields': rows, 'omission_links': marginal,
            'eligible_fields': split_metric['eligible_total'], 'split_correct': split_metric['correct'],
            'broadcast_correct': broadcast_metric['correct'],
            'recovered_fields': broadcast_metric['correct']-split_metric['correct'],
            'single_missing_losses': sum(len(r['missing_ids']) == 1 for r in losses),
            'unique_missing_task_fact_pairs': len(marginal),
            'missing_pairs_affecting_multiple_fields': sum(r['affected_field_count'] > 1 for r in marginal),
            'targeted_rescue_task_fact_pairs': sum(map(len, rescue.values())),
            'targeted_rescue_ids': {str(i): sorted(v) for i, v in rescue.items()}}


def analyze_view(case, tasks, mapping, split_metric, broadcast_metric, status):
    memory = {e['id'] for e in case['entities'] if e['source'] == 'memory'}
    selections = [set(t['keep_entity_ids']) for t in tasks]
    assert len(mapping) == len(tasks) and all(d <= memory for d in selections)
    endpoints = {}
    for endpoint, name in [('lower', 'possibly_needed'), ('upper', 'needed')]:
        needs = reference_sets(case['reference'], memory, name)
        p = partition(selections, mapping, needs)
        for render, metric in [('split', split_metric), ('broadcast', broadcast_metric)]:
            assert p[f'{render}_unnecessary'] == metric[f'L_{endpoint}']
        endpoints[endpoint] = p
    facts = field_analysis(case, tasks, mapping, split_metric, broadcast_metric, status != 'ok')
    assert facts['targeted_rescue_task_fact_pairs'] <= endpoints['upper']['additional_transmissions']
    return {'key': case['key'], 'split': case['split'], 'status': status,
            'active_tasks': sum(t['active'] for t in case['reference']['tasks']),
            'output_blocks': len(tasks), 'mapping': mapping, 'selection_overlap': overlap(selections),
            'reference_overlap': {ep: overlap(list(reference_sets(case['reference'], memory, name).values()))
                                  for ep, name in [('lower', 'possibly_needed'), ('upper', 'needed')]},
            'P_split': split_metric['P_available'], 'P_broadcast': broadcast_metric['P_available'],
            'endpoints': endpoints, 'field_analysis': facts}


def summary(rows):
    if not rows:
        return {'n_inputs': 0}
    result = {'n_inputs': len(rows), 'n_failed': sum(r['status'] != 'ok' for r in rows),
              'P_split': float(np.mean([r['P_split'] for r in rows])),
              'P_broadcast': float(np.mean([r['P_broadcast'] for r in rows]))}
    result['endpoints'] = {}
    for ep in ('lower', 'upper'):
        matrix = np.array([[r['endpoints'][ep]['counts'][k] for k in CATEGORIES] for r in rows])
        totals = matrix.sum(axis=0)
        rng = np.random.default_rng(SEED)
        boot = matrix[rng.integers(0, len(rows), size=(REPEATS, len(rows)))].mean(axis=1)
        result['endpoints'][ep] = {
            'totals': dict(zip(CATEGORIES, map(int, totals))),
            'per_input': dict(zip(CATEGORIES, map(float, totals/len(rows)))),
            'ci95_per_input': {k: np.quantile(boot[:, i], [.025, .975]).tolist() for i, k in enumerate(CATEGORIES)},
            'additional_transmissions_total': int(totals.sum()),
            'unnecessary_fraction': float(totals[1:].sum()/totals.sum()) if totals.sum() else None,
            'global_error_share_of_unnecessary': float(totals[3]/totals[1:].sum()) if totals[1:].sum() else None}
    counts = Counter()
    component = defaultdict(Counter)
    for row in rows:
        counts.update(row['field_analysis']['counts'])
        for f in row['field_analysis']['fields']:
            group = str(f['required_components']) if f['required_components'] < 3 else '3+'
            component[group][f['category']] += 1
    result['field_counts'] = dict(counts)
    result['fields_by_required_components'] = dict(component)
    for k in ('eligible_fields', 'split_correct', 'broadcast_correct', 'recovered_fields',
              'single_missing_losses', 'unique_missing_task_fact_pairs',
              'missing_pairs_affecting_multiple_fields', 'targeted_rescue_task_fact_pairs'):
        result[k] = sum(r['field_analysis'][k] for r in rows)
    return result


def cohort_summary(rows, both_keys=None):
    result = {}
    for split in ('multi', 'single', 'neutral'):
        subset = [r for r in rows if r['split'] == split]
        both = [r for r in subset if both_keys is None or r['key'] in both_keys]
        result[split] = {'all_inputs': summary(subset), 'both_valid': summary(both),
                         'reference_overlap_strata': {ep: {
                             b: summary([r for r in both if bin_overlap(r['reference_overlap'][ep]) == b])
                             for b in ('0-.25', '.25-.5', '.5-.75', '.75-1', 'undefined')}
                             for ep in ('lower', 'upper')},
                         'reference_overlap_by_task_count': {str(m): {
                             ep: dict(Counter(bin_overlap(r['reference_overlap'][ep]) for r in subset if r['active_tasks'] == m))
                             for ep in ('lower', 'upper')} for m in sorted({r['active_tasks'] for r in subset})},
                         'reference_overlap_results_by_task_count': {str(m): {
                             ep: {b: summary([r for r in both if r['active_tasks']==m and bin_overlap(r['reference_overlap'][ep])==b])
                                  for b in ('0-.25','.25-.5','.5-.75','.75-1','undefined')}
                             for ep in ('lower','upper')} for m in sorted({r['active_tasks'] for r in subset})}}
    return result


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_report(result):
    lines = ['# 错误来源、需求共享程度与关键字段损失：缓存诊断', '',
             '日期：2026-09-12。全部数字由现有本地缓存重算，没有新增模型请求。',
             '主分析直接保留等级 3；原结构实验另列。两组指令、任务和请求不同，不能把两组差值解释为等级规则的因果效果。', '',
             '## 主要发现', '',
             '原结构实验的广播负担不能全部解释为正确选择后的需求冲突；其中有相当部分来自全局非必要事实的扩散。当前等级 3 视图以正确选择后的额外分发为主。下表分别在每组的三模型联合路径全输入上汇总分量占比的最小值与最大值；范围跨模型和两个参考端点，不是置信区间。', '',
             '| 实验 | 正确选择外溢占非必要新增 | 全局非必要外溢占非必要新增 |',
             '|---|---:|---:|']
    for cohort, models in result['cohorts'].items():
        shares = defaultdict(list)
        for methods in models.values():
            for e in methods['joint']['multi']['all_inputs']['endpoints'].values():
                denom = e['additional_transmissions_total']-e['totals']['omission_recovery']
                for k in ('needed_selected_spillover','globally_unneeded_spillover'):
                    shares[k].append(e['totals'][k]/denom)
        span = lambda k:f'{100*min(shares[k]):.1f}%–{100*max(shares[k]):.1f}%'
        lines.append(f"| {cohort} | {span('needed_selected_spillover')} | {span('globally_unneeded_spillover')} |")
    lines += ['', '这说明选择误判和分发结构都值得测量；它不证明任何新筛选策略优于旧策略。以下字段与共享程度分析均以等级 3 为主，原结构结果完整保存在同目录。', '',
             '## 定义与解释边界', '',
             '广播新增的每个记录—接收者对被互斥分为四类：', '',
             '1. 遗漏补回：接收者参考需要，但其独立视图未保留。',
             '2. 正确选择外溢：接收者不需要，且至少一个需要者实际选中了该事实。',
             '3. 错投事实外溢：接收者不需要；参考中有需要者，但所有实际选择者都不是需要者。',
             '4. 全局非必要外溢：所有有效任务都不需要，模型仍选择后经广播扩散。', '',
             '后三类之和等于非必要新增传输。四分解是固定参考下的集合归类，并非独立识别模型内部错误原因的因果实验。',
             'lower 使用两自动参考的必要集合并集，upper 使用交集；它们是披露端点，不是各分解分量的上下界。分类随参考改变，逐端点分别报告，不能混搭。',
             '参考共享程度只取有效任务的 Memory 必要集合，rho=(各集合大小之和−并集大小)/((任务数−1)×并集大小)。单任务或空并集单列未定义；与模型选择重叠严格区分。',
             '“关键字段”仅指固定合同要求完整组成的字段，并无人工业务重要性或敏感度评级。一个记录缺失可影响多个字段；记录次数、字段数量、真实任务成功不能互换。', '',
             '## 多任务新增传输分解', '',
             '下表使用全输入，数值为每输入均值，括号为下端点／上端点参考。失败不交付，因此需结合失败数和共同成功表阅读。', '',
             '| 实验 | 模型/方法 | n/失败 | 新增传输 | 遗漏补回 | 正确选择外溢 | 错投事实外溢 | 全局非必要外溢 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for cohort, models in result['cohorts'].items():
        for model, methods in models.items():
            for method, splits in methods.items():
                s = splits['multi']['all_inputs']
                span = lambda k: ' / '.join(f"{s['endpoints'][ep]['per_input'][k]:.3f}" for ep in ('lower', 'upper'))
                add = s['endpoints']['upper']['additional_transmissions_total']/s['n_inputs']
                lines.append(f"| {cohort} | {NAMES[model]}/{method} | {s['n_inputs']}/{s['n_failed']} | {add:.3f} | " + ' | '.join(span(k) for k in CATEGORIES) + ' |')
    lines += ['', '## 等级 3：共同成功输入的字段损失', '',
              '| 模型/方法 | n | 来源不可评分 | 固定请求端点缺口 | 筛选新增丢失 | 其中仅缺一条事实 | 广播补回字段 | 参考引导定向补回记录—任务对 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model, methods in result['cohorts']['required_only'].items():
        for method, splits in methods.items():
            s = splits['multi']['both_valid']; c = s['field_counts']
            lines.append(f"| {NAMES[model]}/{method} | {s['n_inputs']} | {c.get('source_unscorable',0)} | {c.get('fixed_request_endpoint_gap',0)} | {c.get('memory_selection_loss',0)} | {s['single_missing_losses']} | {s['recovered_fields']} | {s['targeted_rescue_task_fact_pairs']} |")
    joint = [result['cohorts']['required_only'][m]['joint']['multi']['both_valid'] for m in MODELS]
    joined = lambda get: '、'.join(str(get(s)) for s in joint)
    lines += ['', f"联合路径中，Qwen、GLM、GPT 的筛选新增字段丢失分别为 {joined(lambda s:s['field_counts'].get('memory_selection_loss',0))} 个，其中仅缺一条事实的分别为 {joined(lambda s:s['single_missing_losses'])} 个；广播仅补回 {joined(lambda s:s['recovered_fields'])} 个完整字段（GPT 限共同成功 {joint[2]['n_inputs']} 条）。这提示大部分剩余遗漏并不能靠共享其他任务已选事实解决。",
              f"“一条缺失事实影响多个字段”在三模型联合路径分别有 {joined(lambda s:s['missing_pairs_affecting_multiple_fields'])} 个记录—任务对，本轮数量很少；不得把这个机制当作已验证的普遍解释。", '',
              '## 完整组成要求与字段损失', '',
              '以下仅含来源可评分且全 Memory、固定请求端点可覆盖的字段，分母为保留字段加筛选新增丢失字段；按字段组成数量分组。GPT 仍取双方成功输入。', '',
              '| 模型/联合 | 1 条组成：丢失/可覆盖 | 2 条组成：丢失/可覆盖 | 至少 3 条组成：丢失/可覆盖 |',
              '|---|---:|---:|---:|']
    for model, methods in result['cohorts']['required_only'].items():
        groups = methods['joint']['multi']['both_valid']['fields_by_required_components']
        cells=[]
        for size in ('1','2','3+'):
            c=groups.get(size,{})
            lost=c.get('memory_selection_loss',0);denom=lost+c.get('retained',0)
            cells.append(f'{lost}/{denom} ({100*lost/denom:.1f}%)' if denom else '0/0')
        lines.append(f"| {NAMES[model]} | " + ' | '.join(cells)+' |')
    lines += ['', '多组成字段的完整保留更脆弱；但完整集合评分本身就要求所有组成同时到达，且不同字段类型、主体和来源构成并未匹配。这是描述性关联，不能据此断言模型存在新的复杂度效应。字段嵌套于输入，不能把表中字段当成相互独立样本。']
    lines += ['', '定向补回计数取每个被广播完整补回字段的缺失记录集合，再按任务去重。它使用隐藏参考，仅是完成同一批固定字段的最小事实集合诊断，不是实现了新的补回算法，也不是端到端效果。多个输出块映射同一任务时，该指标按任务去重，传输指标仍按输出块计。', '',
              '## 参考需求共享程度分布', '',
              '以下分布由输入参考决定，与模型或失败无关；任务数交叉表见 summary.json，字段组成分层见 field_components.csv。', '',
              '| 集合 | 参考端点 | [0,.25) | [.25,.5) | [.5,.75) | [.75,1] | 未定义 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for split, eps in result['reference_distribution'].items():
        for ep, counts in eps.items():
            lines.append(f'| {split} | {ep} | ' + ' | '.join(str(counts.get(b,0)) for b in ('0-.25','.25-.5','.5-.75','.75-1','undefined')) + ' |')
    high_count=result['reference_distribution']['neutral']['upper'].get('.75-1',0)
    lines += ['', f'多任务集合缺乏高共享参考输入，但同场景集合存在 {high_count} 条高共享输入。因此“模型选择视图没有高重叠”不能替代“参考需求没有高共享”的结论。', '',
              '同场景集合的联合路径、上端点参考分层如下；每格为样本数／每输入新增传输均值。GPT 使用双方成功子集，其分母与其他模型不同。', '',
              '| 模型 | [0,.25) | [.25,.5) | [.5,.75) | [.75,1] |',
              '|---|---:|---:|---:|---:|']
    for model, methods in result['cohorts']['required_only'].items():
        bins=methods['joint']['neutral']['reference_overlap_strata']['upper']
        cells=[]
        for b in ('0-.25','.25-.5','.5-.75','.75-1'):
            s=bins[b]
            cells.append(f"{s['n_inputs']} / {sum(s['endpoints']['upper']['per_input'].values()):.2f}" if s['n_inputs'] else '0 / —')
        lines.append(f'| {NAMES[model]} | '+' | '.join(cells)+' |')
    lines += ['', f'同场景集合中，参考共享程度较高的组新增传输较少；高共享组仍可能传输全局非必要事实。各组任务数量、Memory 规模和来源质量不同，不能将该趋势解释为共享程度的独立因果效应。按任务数进一步交叉分层的全部结果见 summary.json 的 reference_overlap_results_by_task_count。高共享仅 {high_count} 条，不据此给出部署阈值。']
    lines += ['', '各共享区间的传输、字段补回与损失见 overlap_strata.csv（共同成功条件分析），同时保留两套参考端点。分层不是需求共享的因果干预，不按该表选择部署界限；小组样本不足时不作趋势或泛化声明。', '',
              '## 可复核案例', '',
              'examples.json 按固定排序选择三类案例：全局非必要外溢、一个缺失记录关联多个字段、广播能补回完整字段。每类最多三条，以等级 3 联合路径为范围；仅用于解释，并非随机代表样本。案例保留任务、字段名和事实 ID，不改写源值或补标必要性。', '',
              '## 对论文贡献的含义', '',
              '可以将贡献推进为：在固定来源参考下，把新增披露区分为正确选择后的额外分发和选择误判的扩散，并结合参考需求共享及完整字段组成说明分发收益的条件。这样的表述比“广播会多传输”更具体，但仍是现有缓存的解释性诊断。',
              '本轮不支持把“一个事实遗漏牵连多个字段”当成主机制，也不能把两个实验之间的分量变化归因为等级规则。更有证据的表述是：完整字段对单个组成遗漏很敏感，而广播通常不能补回大部分筛选遗漏。任何定向补回方法仍需独立设计和验证，当前参考引导计数不是新算法成绩。', '',
              '## 核验与限制', '',
              '- 重算两种分发的暴露和保留，等级 3 拆分逐字段对齐当前主分析；原结构实验对齐冻结评分和原广播增量汇总。',
              '- 逐记录核验四类互斥、总传输守恒、暴露差守恒；逐字段核验广播单调性与定向补回集合。',
              '- 全部已使用输入、原响应、参考、校准和原汇总在运行前后 SHA-256 相同，清单见 source_manifest.json。',
              '- 95% 区间按原输入进行 10000 次 bootstrap，仅反映冻结参考下的输入采样波动。无多重比较校正，不传播标注和重复生成不确定性。',
              '- 原字段来源缺失、接口失败、请求覆盖缺口单列；失败零披露不是优势。没有真人裁决或真实服务效用新增证据。',
              '- 未扫描或校准阈值，未修改主稿、原表、原响应、旧 3.1 结果或人工材料。', '',
              '复现：`python experiments/analyze_error_sources_shared_fields.py`；只读校验：追加 `--check`。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    sources = {}
    def source(path):
        value = read(path)
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    def offline(*a, **kw):
        raise RuntimeError('network_disabled_in_error_source_analysis')
    with patch.object(socket.socket, 'connect', offline), patch.object(socket, 'create_connection', offline):
        raw, cases = dataset('test')
        for p in ('datasets/test.json','datasets/dev.json','datasets/protocol.json','calibration.json','analysis/summary.json'):
            source(FAIR/p)
        source(OUT/'contract_required_only_20260912/summary.json')
        historical_broadcast = source(OUT/'broadcast_diagnostic/summary.json')
        for c in cases:
            a = apply_source_audit(source(OUT/'annotations'/f"{c['key']}.json"), c['item'])
            assert c['entities'] == a['entities'] and c['reference'] == a['reference']
        result = {'version': 'error-sources-shared-fields-1', 'post_hoc': True, 'new_model_calls': 0,
                  'selection_rule': 'grade == 3', 'bootstrap_seed': SEED, 'bootstrap_repeats': REPEATS,
                  'cohorts': {'required_only': {}, 'original_structure': {}}}
        outputs, all_rows = {}, []
        for model in MODELS:
            for c in cases:
                source(FAIR/model/'cases'/f"{c['key']}.json")
            records = load_records(model, cases, raw)
            previous = source(OUT/'contract_required_only_20260912/per_input'/f'{model}.json')
            rows = {m: [] for m in METHODS}
            original = []
            for n, (c, record) in enumerate(zip(cases, records)):
                visible, _ = build_visibility(c)
                for method in METHODS:
                    status = record['method_status'][method]['status']
                    tasks, mapping = required_tasks(c, record[f'{method}_tasks']) if status == 'ok' else ([], [])
                    identity = {id(t): i for t, i in zip(tasks, mapping)}
                    def visibility(t, facts, union):
                        return (set(t['keep_entity_ids']) if union is None else union) | visible[identity[id(t)]]
                    split = score_visible(tasks, mapping, c['entities'], c['reference'], visibility)
                    broadcast = score_visible(tasks, mapping, c['entities'], c['reference'], visibility, broadcast=True)
                    old = previous['metrics'][method][n]
                    assert old['key'] == c['key']
                    for k, v in split.items():
                        assert old[k] == json.loads(json.dumps(v)), (model,method,c['key'],k)
                    row = analyze_view(c, tasks, mapping, split, broadcast, status)
                    for category, count in previous['field_diagnosis'][method][n]['counts'].items():
                        assert row['field_analysis']['counts'].get(category,0) == count
                    rows[method].append(row)
                cached = source(OUT/model/'cases'/f"{c['key']}.json")
                scored = source(OUT/'request_visibility_revision'/model/f"{c['key']}.json")
                original.append(analyze_view(c, cached['tasks'], cached['mapping'],
                                             scored['metrics']['fixed_split'], scored['metrics']['fixed_broadcast'], 'ok'))
            both = {r['key'] for r in records if all(r['method_status'][m]['status'] == 'ok' for m in METHODS)}
            result['cohorts']['required_only'][model] = {m: cohort_summary(rows[m], both) for m in METHODS}
            result['cohorts']['original_structure'][model] = {'joint': cohort_summary(original)}
            for sp in ('multi','single','neutral'):
                got = result['cohorts']['original_structure'][model]['joint'][sp]['all_inputs']
                old = historical_broadcast['models'][model]['splits'][sp]
                assert got['endpoints']['upper']['additional_transmissions_total'] == old['total_added']
                for i, ep in enumerate(('lower','upper')):
                    e = got['endpoints'][ep]
                    assert e['additional_transmissions_total']-e['totals']['omission_recovery'] == old['total_unnecessary'][i]
            for cohort, methods in [('required_only',rows),('original_structure',{'joint':original})]:
                outputs[f'per_input/{cohort}_{model}.json'] = methods
                for method, rr in methods.items():
                    all_rows.extend({'cohort':cohort,'model':model,'method':method,**r} for r in rr)
            print(f'{model}: 350 grade-3 records and 350 structural caches verified', flush=True)
        result['reference_distribution'] = {sp: {ep: dict(Counter(bin_overlap(r['reference_overlap'][ep])
            for r in original if r['split']==sp)) for ep in ('lower','upper')} for sp in ('multi','single','neutral')}
        # 确定性案例选择，所有模型保留；字段和 ID 来自已冻结参考。
        example_rows = [r for r in all_rows if r['cohort']=='required_only' and r['method']=='joint' and r['split']=='multi' and r['status']=='ok']
        examples = {}
        predicates = {'global_spillover': lambda r:r['endpoints']['upper']['counts']['globally_unneeded_spillover']>0,
                      'one_record_multiple_fields':lambda r:r['field_analysis']['missing_pairs_affecting_multiple_fields']>0,
                      'broadcast_field_recovery':lambda r:r['field_analysis']['recovered_fields']>0}
        for name, predicate in predicates.items():
            examples[name] = sorted((r for r in example_rows if predicate(r)), key=lambda r:(r['key'],r['model']))[:3]
        case_map = {c['key']:c for c in cases}
        for name, rr in examples.items():
            enriched=[]
            for r in rr:
                c=case_map[r['key']]
                ids={eid for f in r['field_analysis']['fields'] if f['category']=='memory_selection_loss' for eid in f['missing_ids']}
                if name=='global_spillover':
                    ids.update(eid for _,eid in r['endpoints']['upper']['pairs']['globally_unneeded_spillover'])
                enriched.append({**r,'task_names':{str(t['task_index']):t['name'] for t in c['fixed_tasks']},
                                 'referenced_facts':[e for e in c['entities'] if e['id'] in ids]})
            examples[name]=enriched
        outputs['examples.json'] = examples
        outputs['summary.json'] = result
        outputs['source_manifest.json'] = sources
        outputs['report.md'] = render_report(result)
        flat, components, strata = [], [], []
        for r in all_rows:
            base = {k:r[k] for k in ('cohort','model','method','key','split','status')}
            for ep in ('lower','upper'):
                flat.append({**base,'reference_endpoint':ep,'active_tasks':r['active_tasks'],
                             'reference_overlap':r['reference_overlap'][ep], 'selection_overlap':r['selection_overlap'],
                             **r['endpoints'][ep]['counts'],'recovered_fields':r['field_analysis']['recovered_fields'],
                             'selection_lost_fields':r['field_analysis']['counts'].get('memory_selection_loss',0)})
        for cohort, models in result['cohorts'].items():
            for model, methods in models.items():
                for method, splits in methods.items():
                    for sp, s in splits.items():
                        for pop in ('all_inputs','both_valid'):
                            for size, counts in s[pop]['fields_by_required_components'].items():
                                for category, n in counts.items():
                                    components.append(dict(cohort=cohort,model=model,method=method,split=sp,population=pop,required_components=size,category=category,n_fields=n))
                        for ep, bins in s['reference_overlap_strata'].items():
                            for name, stat in bins.items():
                                row = dict(cohort=cohort,model=model,method=method,split=sp,reference_endpoint=ep,stratum=name,n_inputs=stat['n_inputs'])
                                if stat['n_inputs']:
                                    row.update(P_split=stat['P_split'],P_broadcast=stat['P_broadcast'],recovered_fields=stat['recovered_fields'],selection_lost_fields=stat['field_counts'].get('memory_selection_loss',0),**stat['endpoints'][ep]['per_input'])
                                strata.append(row)
        outputs['per_input.csv'] = csv_text(flat,list(flat[0]))
        outputs['field_components.csv'] = csv_text(components,list(components[0]))
        outputs['overlap_strata.csv'] = csv_text(strata,list(dict.fromkeys(k for r in strata for k in r)))
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items()), 'source_changed'
        for name, value in outputs.items():
            content = value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
            path = DEST/name
            if args.check:
                assert path.read_text(encoding='utf-8') == content, f'changed_artifact:{name}'
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(content,encoding='utf-8')
        print(f'{"Checked" if args.check else "Wrote"} {len(outputs)} artifacts; {len(sources)} source hashes unchanged; network disabled')


if __name__ == '__main__':
    main()

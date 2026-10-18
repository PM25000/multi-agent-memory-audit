"""冻结分歧处理规则，离线复核参考变更对 P、L 与广播分解的影响。

此分析是事后 LLM 规则敏感性，不产生人类答案，不覆盖任何旧参考或结果。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import socket
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analyze_broadcast_additions import aggregate as aggregate_added, decompose
from analyze_fair_comparison_ids import build_visibility, dataset, load_records, paired_ci
from analyze_review_revision import aggregate
from review_fair_comparison_ids import materialize_tasks
from review_llm_adjudication_followup import DECISIONS
from review_request_visibility import score_visible

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/results/review_revision_20260910'
DEST = BASE / 'llm_reference_sensitivity_20260912'
MODELS = ('qwen', 'glm', 'gpt55_responses')
LABELS = ('Qwen3.6-27B', 'GLM5.2', 'GPT-5.5')
VARIANTS = ('original', 'llm_policy', 'interpretation_envelope')
SPLITS = ('multi', 'single', 'neutral')
SCORE_KEYS = ('correct', 'total', 'eligible_total', 'P_available', 'P_all',
              'L_lower', 'L_upper', 'L_all_lower', 'L_all_upper',
              'memory_transmissions', 'unique_L_lower')
READS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    READS[path.relative_to(ROOT).as_posix()] = sha(path)
    return value


def write(path, value, check=False):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if check:
        if path.read_text(encoding='utf-8') != text:
            raise ValueError('artifact_mismatch:' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')


def protocol():
    private = read(BASE / 'human_validation/scoring_private.json')
    packet = read(BASE / 'human_adjudication_followup/packet.json')
    changes = []
    # 对先前 uncertain 的四项明确本轮操作定义；不倒改初次复核的记录。
    resolved = {
        ('P038', 'presence:1:e22'): ('absent', [],
            '出现性主标签要求当前用途/关系成立；另记住宿地点属性已披露，不将属性出现冒充当前请求命题出现。'),
        ('S061', 'binding:1:偏好'): ('available', ['e16', 'e22', 'e23', 'e24', 'e25'],
            '景点偏好包括明确景点特点、目的地体验和指定兴趣；排除饮食耐受、行程节奏以及“推荐景点”指令本身。这是本轮字段范围定义。'),
        ('S099', 'binding:2:电商账号'): ('missing', [],
            '账号字段要求明确账号值，绑定手机号只有在合同明确允许作为账号标识时才替代；当前合同未给该许可。'),
        ('S268', 'binding:0:使用场景'): ('available', ['e26'],
            '使用场景按当前请求直接表述绑定；更窄的历史背景和未来评价维度不强制作为当前字段的额外 AND 组成。'),
    }
    for prior in DECISIONS:
        row = copy.deepcopy(prior)
        fid, key = row['form_id'], row['key']
        if (fid, key) in resolved:
            row['prior_verdict'] = row['verdict']
            row['verdict'], row['entity_ids'], row['operational_rule'] = resolved[(fid, key)]
        if key.startswith('binding:'):
            meta = private['forms'][fid]
            row['case_key'] = meta['case_key']
            _, ti, field = key.split(':', 2)
            row.update(task_index=int(ti), field=field)
            original = next(b for b in meta['annotation']['reference']['bindings']
                            if b['task_index'] == int(ti) and b['field'] == field)
            pending = next(r for r in packet['items'] if r['form_id'] == fid and r['key'] == key)
            alternatives = [original['a'], original['b'], pending['a'], pending['b'], row]
            row['candidate_necessary_sets'] = [list(ids) for ids in sorted({
                tuple(sorted(a.get('entity_ids', []))) if a.get('status', a.get('verdict')) == 'available' else ()
                for a in alternatives})]
        changes.append(row)
    return dict(version='llm-reference-policy-1', review_type='post_hoc_LLM_rule_sensitivity',
                reviewer='Codex 当前会话', human_attestation=False, blind=False,
                rules=[
                    '只更新17项复核对应的10个字段绑定；其他字段保持冻结参考。',
                    '当前任务的主体、时间和用途必须有直接证据；相关背景不能填补缺失值。',
                    '不同所需值为AND组成；同一当前值的重复来源不额外强制保留，优先当前请求直接证据。',
                    '字段语义未定时明确本轮操作定义，同时保留全部原有解释作敏感性范围。',
                    '披露范围在每任务合并所有字段贡献后计算，不能因一个字段改判就删除另一字段仍需要的记录。',
                    '候选漏项只登记，不将新事实偷偷加入模型未见的候选表或补造模型输出。',
                    '只重评分，冻结载荷、请求出现性、任务映射、调用和3.1阈值；不重选工作点。',
                    '解释范围允许原两模型、人A/B和本轮规则的所有候选集合，空集合也保留。',
                ], changes=changes, source_hashes=dict(READS))


def revise_reference(reference, updates, envelope=False):
    """字段贡献先按任务合并；保留其他字段对同一事实的独立需求。"""
    result = copy.deepcopy(reference)
    updates = {(u['task_index'], u['field']): u for u in updates}
    if not updates:
        return result
    a, b = defaultdict(set), defaultdict(set)
    certain, possible = defaultdict(set), defaultdict(set)
    seen = set()
    for field in result['bindings']:
        ti = field['task_index']
        key = ti, field['field']
        if key not in updates:
            a[ti].update(field['a']['entity_ids'])
            b[ti].update(field['b']['entity_ids'])
            continue
        update = updates[key]
        seen.add(key)
        field['llm_override'] = dict(status=update['verdict'], entity_ids=update['entity_ids'])
        field['status'], field['entity_ids'] = update['verdict'], list(update['entity_ids'])
        # 原 a/b 与 agreement 原样保留，仅表示历史两模型情况；本轮没有伪造共识。
        sets = [set(ids) for ids in update['candidate_necessary_sets']] if envelope else [set(update['entity_ids'])]
        certain[ti].update(set.intersection(*sets))
        possible[ti].update(set.union(*sets))
    assert seen == set(updates)
    result['needed'] = {str(t['task_index']): sorted((a[t['task_index']] & b[t['task_index']]) | certain[t['task_index']])
                        for t in result['tasks']}
    result['possibly_needed'] = {str(t['task_index']): sorted(a[t['task_index']] | b[t['task_index']] | possible[t['task_index']])
                                 for t in result['tasks']}
    result['reference_provenance'] = 'LLM policy with original unreviewed fields; not new dual-model consensus'
    result['necessity_mode'] = 'candidate-interpretation-envelope' if envelope else 'specified-LLM-policy'
    return result


def rescore(visibility, entities, reference):
    """只用冻结的实际可见记录；不重新判断请求出现性或重写载荷。"""
    memory = {e['id'] for e in entities if e['source'] == 'memory'}
    valid_ids = {e['id'] for e in entities}
    bytask = defaultdict(set)
    lower, upper, all_lower, all_upper, transmissions = [], [], [], [], 0
    active = {t['task_index']: t['active'] for t in reference['tasks']}
    for block in visibility:
        ti, bi, seen = block['task_index'], block['block_index'], set(block['visible'])
        assert seen <= valid_ids
        bytask[ti].update(seen)
        needed = set(reference['needed'].get(str(ti), [])) if active.get(ti, False) else set()
        possible = set(reference['possibly_needed'].get(str(ti), [])) if active.get(ti, False) else set()
        assert needed <= possible <= valid_ids
        lower += [(bi, e) for e in (seen & memory) - possible]
        upper += [(bi, e) for e in (seen & memory) - needed]
        all_lower += [(bi, e) for e in seen - possible]
        all_upper += [(bi, e) for e in seen - needed]
        transmissions += len(seen & memory)
    fields = []
    for f in reference['bindings']:
        eligible = f['status'] == 'available' and active[f['task_index']]
        correct = eligible and set(f['entity_ids']) <= bytask[f['task_index']]
        fields.append(dict(task_index=f['task_index'], field=f['field'], eligible=eligible,
                           correct=bool(correct), reference_status=f['status']))
    correct, eligible = sum(f['correct'] for f in fields), sum(f['eligible'] for f in fields)
    return dict(correct=correct, total=len(fields), eligible_total=eligible,
                P_available=correct / eligible if eligible else None, P_all=correct / len(fields),
                L_lower=len(lower), L_upper=len(upper), L_all_lower=len(all_lower), L_all_upper=len(all_upper),
                memory_transmissions=transmissions, unique_L_lower=len({e for _, e in lower}), fields=fields,
                visibility=visibility, leak_pairs_lower=sorted(lower), leak_pairs_upper=sorted(upper))


def pair_summary(a, b):
    return {k: paired_ci([m[k] for m in a], [m[k] for m in b]) for k in ('P_available', 'L_lower', 'L_upper')}


def metrics_equal(a, b):
    assert all(a[k] == b[k] for k in SCORE_KEYS), [(k, a[k], b[k]) for k in SCORE_KEYS if a[k] != b[k]]
    assert a['fields'] == b['fields']
    for name in ('leak_pairs_lower', 'leak_pairs_upper'):
        assert set(map(tuple, a[name])) == set(map(tuple, b[name]))


def frozen_fair_metric(case, record, method):
    if record['method_status'][method]['status'] != 'ok':
        return score_visible([], [], case['entities'], case['reference'], lambda *args: set())
    tasks, mapping = materialize_tasks(case, record[method + '_tasks'], 3.1)
    visible, _ = build_visibility(case)
    ids = {id(t): i for t, i in zip(tasks, mapping)}
    return score_visible(tasks, mapping, case['entities'], case['reference'],
                         lambda t, f, u: set(t['keep_entity_ids']) | visible[ids[id(t)]])


def run(check=False):
    policy_path = DEST / 'protocol.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    assert policy == protocol(), 'frozen_policy_or_source_changed'
    READS[policy_path.relative_to(ROOT).as_posix()] = sha(policy_path)
    protected = [p for folder in ('human_validation', 'human_adjudication_followup', 'llm_adjudication_followup_20260912')
                 for p in (BASE / folder).rglob('*') if p.is_file()]
    protected_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    raw_dataset = read(BASE / 'fair_comparison_ids/datasets/test.json')
    cases = raw_dataset['cases'] if isinstance(raw_dataset, dict) else raw_dataset
    assert len(cases) == 350 and len({c['key'] for c in cases}) == 350
    updates = defaultdict(list)
    for row in policy['changes']:
        if row['key'].startswith('binding:'):
            updates[row['case_key']].append(row)
    assert sum(map(len, updates.values())) == 10
    references = {}
    for case in cases:
        references[case['key']] = {'original': case['reference'],
            'llm_policy': revise_reference(case['reference'], updates[case['key']]),
            'interpretation_envelope': revise_reference(case['reference'], updates[case['key']], True)}
    # 匹配固定短版选中的10条来源输入，最坏情况允许这些输入的全部必要性标签失效。
    private = read(BASE / 'human_validation/scoring_private.json')
    plan = read(BASE / 'human_validation/hour_plan_private.json')
    audit_keys = {private['forms'][fid]['case_key'] for fid in plan['selected_checks'] if fid.startswith('S')}
    assert len(audit_keys) == 10
    data, structure_summary, fair_summary, stress = {}, {}, {}, {}
    read(BASE / 'fair_comparison_ids/calibration.json')
    read(BASE / 'fair_comparison_ids/datasets/protocol.json')
    for model in MODELS:
        print('Analyzing ' + model, flush=True)
        groups = {v: [] for v in VARIANTS}
        fair_groups = {v: [] for v in VARIANTS}
        records = load_records(model, cases, raw_dataset)
        for case, fair_record in zip(cases, records):
            key = case['key']
            read(BASE / 'fair_comparison_ids' / model / 'cases' / (key + '.json'))
            raw = read(BASE / model / 'cases' / (key + '.json'))
            cached = read(BASE / 'request_visibility_revision' / model / (key + '.json'))
            assert raw['key'] == cached['key'] == key
            fair_original = {m: frozen_fair_metric(case, fair_record, m) for m in ('joint', 'independent')}
            for variant in VARIANTS:
                reference = references[key][variant]
                metrics = {condition: rescore(cached['metrics'][condition]['visibility'], case['entities'], reference)
                           for condition in ('fixed_split', 'fixed_broadcast')}
                if variant == 'original':
                    for condition, value in metrics.items():
                        metrics_equal(value, cached['metrics'][condition])
                addition = decompose(raw['tasks'], metrics['fixed_split'], metrics['fixed_broadcast'])
                groups[variant].append(dict(key=key, split=case['split'], metrics=metrics, addition=addition))
                fair_metrics = {m: rescore(fair_original[m]['visibility'], case['entities'], reference)
                                for m in ('joint', 'independent')}
                if variant == 'original':
                    for method in fair_metrics:
                        metrics_equal(fair_metrics[method], fair_original[method])
                fair_groups[variant].append(dict(key=key, split=case['split'], metrics=fair_metrics,
                    method_status=fair_record['method_status'],
                    both_valid=all(fair_record['method_status'][m]['status'] == 'ok' for m in fair_metrics)))
        structure_summary[model], fair_summary[model] = {}, {}
        for variant in VARIANTS:
            structure_summary[model][variant], fair_summary[model][variant] = {}, {}
            for split in SPLITS:
                rows = [r for r in groups[variant] if r['split'] == split]
                assert len(rows) == (50 if split == 'single' else 150)
                structure_summary[model][variant][split] = {
                    'conditions': {c: aggregate([r['metrics'][c] for r in rows]) for c in ('fixed_split', 'fixed_broadcast')},
                    'broadcast_minus_split': pair_summary([r['metrics']['fixed_broadcast'] for r in rows],
                                                         [r['metrics']['fixed_split'] for r in rows]),
                    'broadcast': aggregate_added([r['addition'] for r in rows]),
                }
                fr = [r for r in fair_groups[variant] if r['split'] == split]
                fair_summary[model][variant][split] = {}
                for population, selected in [('all_inputs', fr), ('both_valid', [r for r in fr if r['both_valid']])]:
                    fair_summary[model][variant][split][population] = dict(n=len(selected),
                        methods={m: aggregate([r['metrics'][m] for r in selected]) for m in ('joint', 'independent')},
                        joint_minus_independent=pair_summary([r['metrics']['joint'] for r in selected],
                                                            [r['metrics']['independent'] for r in selected]))
        # 范围必须包住原参考和新规则，不以范围缩小掩盖不确定性。
        for old, new, envelope in zip(groups['original'], groups['llm_policy'], groups['interpretation_envelope']):
            assert old['addition']['additional_transmissions'] == new['addition']['additional_transmissions']
            for c in ('fixed_split', 'fixed_broadcast'):
                for point in (old, new):
                    assert envelope['metrics'][c]['L_lower'] <= point['metrics'][c]['L_lower']
                    assert envelope['metrics'][c]['L_upper'] >= point['metrics'][c]['L_upper']
        multi = [r for r in groups['original'] if r['split'] == 'multi']
        added = sum(r['addition']['additional_transmissions'] for r in multi)
        outside = [r for r in multi if r['key'] not in audit_keys]
        lower = sum(r['addition']['unnecessary_lower'] for r in outside)
        lower_original = sum(r['addition']['unnecessary_lower'] for r in multi)
        stress[model] = dict(n_audited_multi=len(multi) - len(outside), total_added=added,
            description='Allow all added pairs in the 5 source-audited multi inputs to be necessary; other inputs fixed.',
            lower_fraction_if_all_audited_added_pairs_needed=lower / added,
            label_flips_to_below_95_percent=max(0, math.floor(lower_original - .95 * added) + 1),
            label_flips_to_below_90_percent=max(0, math.floor(lower_original - .90 * added) + 1),
            label_flips_unit='previously unnecessary added record-recipient pairs, not unique facts or input count')
        data[model] = {'structure': groups, 'fair_locked_3_1': fair_groups}
    # 冻结输入的哈希再次检查；分析产物单独保存。
    assert all(sha(ROOT / p) == h for p, h in READS.items())
    assert all(sha(ROOT / p) == h for p, h in protected_hashes.items())
    summary = dict(version=policy['version'], network_calls=0, structure=structure_summary,
                   fair_locked_3_1=fair_summary, adversarial_stress=stress,
                   scope=dict(reviewed_checks=17, reviewed_bindings=10, binding_inputs=len([k for k,v in updates.items() if v]),
                              source_audited_inputs=10, rescored_inputs_per_model=350,
                              fresh_annotation_of_all_inputs=False, human_validation_complete=False),
                   source_sha256=dict(READS), protocol_sha256=sha(policy_path))
    write(DEST / 'references.json', references, check)
    for model in MODELS:
        write(DEST / (model + '_rescored.json'), data[model], check)
    write(DEST / 'summary.json', summary, check)
    write(DEST / 'qa.json', dict(original_metrics_reproduced=True, cases_per_model=350, models=3,
          interpretation_encloses_original_and_llm=True, transmission_counts_unchanged=True,
          protected_sha256=protected_hashes, analysis_script_sha256=sha(Path(__file__))), check)
    render_report(summary, policy, check)
    print(json.dumps({'verified' if check else 'written': True, 'scope': summary['scope'],
                      'multi_fractions': {m: {v: structure_summary[m][v]['multi']['broadcast']['unnecessary_fraction']
                                             for v in VARIANTS} for m in MODELS}, 'stress': stress}, ensure_ascii=False))


def span(values, scale=1, digits=2):
    return '--'.join(f'{x * scale:.{digits}f}' for x in values)


def render_report(summary, policy, check):
    lines = ['# 人工抽检分歧：LLM 规则修订与主结论敏感性', '',
             '本轮完成的是自动参考修订与敏感性检验，不是新增真人裁决。规则在本轮重评分前单独冻结，但已知旧结果及 A/B 答案，属于事后分析。', '',
             '## 范围与规则', '',
             '17项逐项形成操作标签；其中此前4项不确定通过明确规则获得条件性标签，不代表原语义歧义消失。更新7条输入的10个字段绑定，其余参考不变；重放三模型各350条结构实验与固定合同判断。', '',
             '保留原参考、新规则和解释范围三个版本。解释范围纳入原两模型、A/B及新规则的必要集合；它不是可信区间，也不覆盖未知共同错误。范围版的P沿用新规则字段，只有L及新增传输组成取范围。', '']
    lines += ['- ' + r for r in policy['rules']]
    lines += ['', '## 规则更新', '', '| 项目 | 本轮标签 | 明确的规则或理由 |', '|---|---|---|']
    for row in policy['changes']:
        lines.append(f"| {row['form_id']} / {row['key']} | {row['verdict']} {','.join(row['entity_ids'])} | {row.get('operational_rule',row['reason'])} |")
    lines += ['', '## 多任务结构结果', '',
              '| 模型 | 参考 | 独立P / 广播P（%） | 独立L | 广播L | 新增传输非必要占比（%） |',
              '|---|---|---|---|---|---|']
    tex = [r'\subsection{LLM 复核后的参考敏感性}', r'\label{app:llm-reference-sensitivity}',
           '对有限双人抽检中的17项分歧或不确定项，我们进行事后LLM复核。此前4项未定通过明确操作定义获得条件标签；10个字段绑定涉及7条输入。原答卷及自动参考保持冻结，新规则仅用于另行重评分。复核已看到人工答案和自动参考，不构成独立人类裁决。',
           '', '规则区分当前需求与历史背景，避免把出发时间当作到达时间、建档医院当作本次急诊医院；账号字段不默认接受绑定手机号。宽泛偏好和用途明确本轮范围，同一当前值的重复来源不额外要求全部到达。5项出现性另行区分属性披露与当前用途命题，2项候选遗漏登记而不补造模型选择。',
           '', '固定各模型350条输入的实际载荷、任务映射和请求出现性，重新计算字段保留、Memory披露及广播新增传输。下表为150条多任务结果。新规则仍使用未复核字段的原参考端点；解释范围额外允许被复核字段采用原两模型、A/B或新规则的任一必要集合。',
           r'\begin{table}[!htbp]\centering\small',
           r'\caption{分歧参考的事后敏感性。P为独立／广播宏平均；L为独立／广播的Memory计数范围。解释范围只作用于L和新增组成，P沿用新规则。}',
           r'\label{tab:llm-reference-sensitivity}',
           r'\begin{tabular}{@{}llrrr@{}}\toprule',
           r'模型 & 参考 & P（\%） & L（独立／广播） & 非必要占比\\\midrule']
    for model, label in zip(MODELS, LABELS):
        for variant, name in zip(VARIANTS, ('原参考', 'LLM规则', '解释范围')):
            s = summary['structure'][model][variant]['multi']
            a, b = (s['conditions'][c] for c in ('fixed_split', 'fixed_broadcast'))
            p = f"{100*a['P_available_macro']:.2f} / {100*b['P_available_macro']:.2f}"
            l1, l2 = span([a['L_lower'], a['L_upper']]), span([b['L_lower'], b['L_upper']])
            fraction = span(s['broadcast']['unnecessary_fraction'], 100)
            lines.append(f'| {label} | {name} | {p} | {l1} | {l2} | {fraction} |')
            tex.append(f'{label} & {name} & {p} & {l1} / {l2} & {fraction}' + r'\% \\')
    tex += [r'\bottomrule\end{tabular}\end{table}', '']
    lines += ['', '各集合的宏/微保留率、分母、完整载荷计数、配对区间和新增比例bootstrap区间见summary.json。只因参考改变而产生的分数变化不代表模型变好。', '',
              '## 固定合同联合／逐任务：原锁定3.1不重选', '',
              '| 模型 | 参考 | 共同有效n | 联合P / 逐任务P（%） | L差及95%区间 |', '|---|---|---:|---|---|']
    for model, label in zip(MODELS, LABELS):
        for variant in ('original', 'llm_policy'):
            s = summary['fair_locked_3_1'][model][variant]['multi']['both_valid']
            j, i = (s['methods'][m] for m in ('joint', 'independent'))
            d = s['joint_minus_independent']['L_upper']
            lines.append(f"| {label} | {variant} | {s['n']} | {100*j['P_available_macro']:.2f} / {100*i['P_available_macro']:.2f} | {d['difference_joint_minus_independent']:+.4f} [{span(d['ci95'])}] |")
    lines += ['', '全输入与共同有效结果分别保留；失败载荷仍为空，不推断缺失费用。调用和token不随本轮参考重评分变化。本轮没有重选3.1，也没有将新的等级3简化分析当作锁定实验。', '',
              '## 对已抽检输入的极端压力检查', '',
              '允许10条来源抽检输入中的全部必要性标签都不可靠；其中5条属于多任务组。将这5条输入所有广播新增对都视为必要，其余输入保持旧参考。此界限用于检查结论对这5条输入的依赖，不代表其他145条已验证。', '',
              '| 模型 | 新增非必要占比下界（%） | 全集合跌破95%所需标签翻转数 | 跌破90%所需翻转数 |',
              '|---|---:|---:|---:|']
    for model, label in zip(MODELS, LABELS):
        s = summary['adversarial_stress'][model]
        lines.append(f"| {label} | {100*s['lower_fraction_if_all_audited_added_pairs_needed']:.2f} | {s['label_flips_to_below_95_percent']} | {s['label_flips_to_below_90_percent']} |")
    lines += ['', '翻转单位是“新增记录—接收者对”，不是独立事实或独立错误；一个系统性字段错误可影响多个对。两套标注范围和bootstrap均不替代未知错误分析。', '',
              '## 仍然成立的边界', '',
              '- 原事实表遗漏没有被偷偷补成模型已见候选。S088地址与S333未订机票状态已登记，但旧缓存不能证明模型面对完整候选时的判断。',
              '- 本轮不是全350条人工/LLM重新标注，更不能证明不存在其他系统性错误。',
              '- 参数生成和SQLite执行结果未重评；不能将参考修订解释为下游任务成功率提升。',
              '- 抽检揭示的问题已有明确规则、可复现重评分和结论影响记录；独立准确性验证及真实分布外推仍属于证据边界。', '',
              '## 复现', '',
              '`python experiments/analyze_llm_reference_sensitivity.py --check`：断开socket，重算并逐字校验新产物；核对旧结构指标完全复现、原参考和LLM结果均被解释范围包住，以及冻结材料哈希未变。', '',
              '[规则](protocol.json) · [参考各版本](references.json) · [统计](summary.json) · [核验](qa.json)', '']
    fractions = [summary['structure'][m]['interpretation_envelope']['multi']['broadcast']['unnecessary_fraction'] for m in MODELS]
    low, high = min(s[0] for s in fractions), max(s[1] for s in fractions)
    stress_low = min(summary['adversarial_stress'][m]['lower_fraction_if_all_audited_added_pairs_needed'] for m in MODELS)
    tex += [f'在上述解释范围内，三模型多任务新增传输的非必要占比为{100*low:.2f}--{100*high:.2f}' + r'\%。',
            f'若进一步将5条来源抽检多任务输入的所有新增传输都计为必要，其余145条保持原参考，三模型该比例的最低下界为{100*stress_low:.2f}' + r'\%。这一压力检查只界定已抽检输入对结果的影响，不证明其余标签正确。', '',
            '固定合同联合与逐任务比较另保持原阈值3.1重算，调用与token不变；各集合及共同有效子集的配对结果随复现材料提供。新参考没有改变模型生成过程，分数变化不作为能力提升。', '',
            '本分析只修订10个被复核字段，不是350条输入的完整重新标注。字段解释范围不覆盖未知共同错误；原事实表漏项也未被补成模型已见候选。故这些结果检验主结论对本轮已识别问题的敏感性，不能替代独立参考准确性验证。', '']
    write(DEST / 'README.md', '\n'.join(lines), check)
    write(DEST / 'appendix_llm_reference.tex', '\n'.join(tex), check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    with patch.object(socket, 'socket', side_effect=RuntimeError('offline analysis: network disabled')):
        if args.freeze:
            value = protocol()
            path = DEST / 'protocol.json'
            if path.exists():
                assert json.loads(path.read_text(encoding='utf-8')) == value, 'existing_protocol_is_immutable'
            else:
                write(path, value)
            print('Protocol frozen before rescoring: ' + sha(path))
        else:
            run(args.check)

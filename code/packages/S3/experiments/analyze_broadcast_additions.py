"""Post-hoc transmission decomposition from frozen caches; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/results/review_revision_20260910'
DEST = BASE / 'broadcast_diagnostic'
MODELS = {'qwen': 'Qwen3.6-27B', 'glm': 'GLM5.2', 'gpt55_responses': 'GPT-5.5'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def decompose(tasks, split, broadcast):
    selections = [set(task['keep_entity_ids']) for task in tasks]
    union = set().union(*selections)
    sent_d = {(i, e) for i, selected in enumerate(selections) for e in selected}
    sent_u = {(i, e) for i in range(len(tasks)) for e in union}
    added = sent_u - sent_d
    expected = sum(len(tasks) - sum(e in selected for selected in selections) for e in union)
    if len(added) != expected:
        raise ValueError('transmission_identity_failed')
    leaks = {}
    for name, score, sent in [('split', split, sent_d), ('broadcast', broadcast, sent_u)]:
        if score['memory_transmissions'] != len(sent):
            raise ValueError('transmission_count_mismatch')
        for endpoint in ('lower', 'upper'):
            pairs = {tuple(pair) for pair in score['leak_pairs_' + endpoint]}
            if len(pairs) != score['L_' + endpoint] or not pairs <= sent:
                raise ValueError('leak_pair_count_or_membership_mismatch')
            leaks[name, endpoint] = pairs
        if not leaks[name, 'lower'] <= leaks[name, 'upper']:
            raise ValueError('necessity_endpoints_not_nested')
    extras = {}
    for endpoint in ('lower', 'upper'):
        old, new = leaks['split', endpoint], leaks['broadcast', endpoint]
        # A fixed reference must assign the same label to an existing pair.
        if not old <= new or new & sent_d != old:
            raise ValueError('reference_changed_between_renderings')
        extras[endpoint] = new - old
        if not extras[endpoint] <= added:
            raise ValueError('added_leak_not_added_transmission')
    return {
        'output_blocks': len(tasks), 'additional_transmissions': len(added),
        'unnecessary_lower': len(extras['lower']), 'unnecessary_upper': len(extras['upper']),
        'necessary_lower': len(added) - len(extras['upper']),
        'necessary_upper': len(added) - len(extras['lower']),
    }


def aggregate(rows, repeats=10000):
    counts = np.array([[r[k] for k in ('additional_transmissions', 'unnecessary_lower', 'unnecessary_upper')]
                       for r in rows], dtype=np.int64)
    totals = counts.sum(axis=0)
    result = {'n_inputs': len(rows), 'n_zero_addition_inputs': int((counts[:, 0] == 0).sum()),
              'total_added': int(totals[0]), 'total_unnecessary': totals[1:].tolist(),
              'per_input_added': float(totals[0] / len(rows)),
              'per_input_unnecessary': (totals[1:] / len(rows)).tolist(),
              'per_input_necessary': ((totals[0] - totals[:0:-1]) / len(rows)).tolist(),
              'unnecessary_fraction': None, 'fraction_ci95_by_endpoint': None}
    if totals[0]:
        result['unnecessary_fraction'] = (totals[1:] / totals[0]).tolist()
        rng = np.random.default_rng(20260911)
        boot = counts[rng.integers(0, len(rows), size=(repeats, len(rows)))].sum(axis=1)
        valid = boot[:, 0] > 0
        result['fraction_ci95_by_endpoint'] = [
            np.quantile(boot[valid, j] / boot[valid, 0], [.025, .975]).tolist() for j in (1, 2)]
        result['zero_denominator_bootstrap_replicates'] = int((~valid).sum())
    return result


def analyze():
    sources, groups = {}, {}
    for model in MODELS:
        rows = []
        for path in sorted((BASE / 'request_visibility_revision' / model).glob('*.json')):
            raw_path = BASE / model / 'cases' / path.name
            for source in (path, raw_path):
                sources[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
            corrected, raw = read(path), read(raw_path)
            if (corrected['key'], corrected['split']) != (raw['key'], raw['split']):
                raise ValueError('raw_and_corrected_input_identity_mismatch')
            row = decompose(raw['tasks'], corrected['metrics']['fixed_split'], corrected['metrics']['fixed_broadcast'])
            rows.append({'key': corrected['key'], 'split': corrected['split'], **row})
        if len(rows) != 350:
            raise ValueError('expected_350_cached_inputs_per_model')
        if len({row['key'] for row in rows}) != 350:
            raise ValueError('duplicate_input_key')
        for split, count in (('multi', 150), ('single', 50), ('neutral', 150)):
            if sum(row['split'] == split for row in rows) != count:
                raise ValueError('unexpected_split_size:' + split)
        groups[model] = {'rows': rows, 'splits': {split: aggregate([r for r in rows if r['split'] == split])
                                               for split in ('multi', 'single', 'neutral')}}
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in sources.items()):
        raise ValueError('source_changed_during_analysis')
    return {'analysis': 'post_hoc_broadcast_added_transmissions_v1', 'models': groups,
            'source_sha256': sources, 'bootstrap_seed': 20260911, 'bootstrap_repeats': 10000,
            'interpretation': 'Ratio of pooled added record-recipient transmissions, not mean input ratios, field recovery, human accuracy, or real task success. Necessity endpoints are frozen model-label sensitivity; endpoint bootstrap intervals resample original inputs and do not include reference uncertainty.'}


def table(result):
    lines = [r'\begin{table}[htbp]\centering\small',
             r'\caption{广播新增传输的事后分解，每模型 150 条多任务输入。前三个数值列为每输入记录--接收者传输数；末列为总非必要增量除以总传输增量。范围来自两套必要性参考，不是置信区间。}',
             r'\label{tab:broadcast-additions}', r'\begin{tabular}{@{}lrrrr@{}}\toprule',
             r'模型 & 新增传输 & 其中必要 & 其中非必要 & 非必要占比 \\ \midrule']
    for model, label in MODELS.items():
        s = result['models'][model]['splits']['multi']
        span = lambda xs: f'{xs[0]:.2f}--{xs[1]:.2f}'
        fraction = span([x * 100 for x in s['unnecessary_fraction']])
        lines.append(f"{label} & {s['per_input_added']:.2f} & {span(s['per_input_necessary'])} & {span(s['per_input_unnecessary'])} & {fraction}\\% \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Recompute and verify existing artifacts without writing.')
    args = parser.parse_args()
    result = analyze()
    outputs = {DEST / 'summary.json': json.dumps(result, ensure_ascii=False, indent=2) + '\n',
               ROOT / 'paper-cn/broadcast_additions_table.tex': table(result)}
    for path, text in outputs.items():
        if args.check:
            if path.read_text(encoding='utf-8') != text:
                raise ValueError('broadcast_diagnostic_changed:' + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
    print(json.dumps({'verified' if args.check else 'written': True, 'cached_inputs': 1050, 'network_calls': 0,
                      'multi': {m: v['splits']['multi'] for m, v in result['models'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()

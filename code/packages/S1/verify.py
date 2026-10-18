"""Standalone numerical replay from scored observations. No network or model SDK."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def interval(values, indices):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'ci95': np.quantile(a[indices].mean(axis=1), [.025, .975]).tolist()}


def check_interval(actual, expected):
    close(actual['mean'], expected['mean'])
    close(actual['ci95'], expected['ci95'])


def verify():
    for name, expected_hash in read('manifest.json')['files_sha256'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected_hash, name
    expected = read('expected.json')
    core, business, probe = read('core.json'), read('business.json'), read('probe.json')
    report = {'scope': 'Numerical replay from frozen scored observations; no semantic validation or new model calls',
              'python': platform.python_version(), 'numpy': np.__version__, 'bootstrap_draws': 10000,
              'core': {}, 'business': {}, 'probe': {}}

    # Preserve original input order and paired rows for exact percentile replay.
    for model, target in expected['core'].items():
        rows = [r for r in core if r['model'] == model]
        assert len(rows) == len({r['key'] for r in rows}) == 150
        assert sum(r['eligible_fields'] for r in rows) == target['eligible_fields'] == 1406
        assert all(r['eligible_fields'] > 0 for r in rows)
        idx = np.random.default_rng(20260912).integers(0, 150, (10000, 150))
        result = {'P': {}, 'endpoints': {}}
        for label, values in [('split', [r['P_split'] for r in rows]), ('broadcast', [r['P_broadcast'] for r in rows]),
                              ('delta', [r['P_broadcast'] - r['P_split'] for r in rows])]:
            result['P'][label] = interval(values, idx)
            check_interval(result['P'][label], target['P'][label])
        for ep in ('lower', 'upper'):
            result['endpoints'][ep] = {}
            for label in ('split_L', 'broadcast_L', 'delta_L'):
                value = interval([r['endpoints'][ep][label] for r in rows], idx)
                check_interval(value, target['endpoints'][ep][label])
                result['endpoints'][ep][label] = value
            categories = ('needed_selected_spillover', 'needed_but_wrong_recipient_spillover', 'globally_unneeded_spillover')
            totals = [sum(r['endpoints'][ep]['spillover_counts'][k] for r in rows) for k in categories]
            denom = sum(r['endpoints'][ep]['delta_L'] for r in rows)
            assert sum(totals) == denom and denom > 0
            st = target['endpoints'][ep]['structure_totals']
            assert totals == [st[k] for k in ('correct_added', 'wrong_added', 'global_added')]
            result['endpoints'][ep]['spillover_percent'] = [100 * n / denom for n in totals]
        report['core'][model] = result

    # All nine conditions, failures retained, and every frozen workflow interval.
    for model, target in expected['business'].items():
        rows = [r for r in business if r['model'] == model]
        bgs = sorted({r['background'] for r in rows})
        assert len(rows) == 216 and len(bgs) == 12
        lookup = {(r['background'], r['sharing'], r['pattern'], r['arm']): r for r in rows}
        assert len(lookup) == len(rows)
        idx = np.random.default_rng(20260915).integers(0, 12, (10000, 12))
        result = {'cells': {}, 'transport': {}}
        for condition, arms in target['cells'].items():
            level, pattern = condition.split('/')
            result['cells'][condition] = {}
            for arm, totals in arms.items():
                part = [lookup[bg, level, pattern, arm] for bg in bgs]
                assert len(part) == totals['workflows'] == 12
                for key in ('workflow_success', 'task_success', 'tasks', 'responses_received', 'format_valid',
                            'transport_unresolved', 'input_forbidden_planned', 'input_forbidden_confirmed',
                            'output_forbidden', 'committed_tool_forbidden', 'not_attempted'):
                    assert sum(r[key] for r in part) == totals[key], (model, condition, arm, key)
                result['cells'][condition][arm] = {k: sum(r[k] for r in part) for k in ('workflow_success', 'task_success')}
            delta = [int(lookup[bg, level, pattern, 'U']['workflow_success']) - int(lookup[bg, level, pattern, 'D']['workflow_success']) for bg in bgs]
            value = interval(delta, idx)
            check_interval(value, target['paired_statistics']['estimates'][condition]['workflow_delta'])
            result['cells'][condition]['workflow_delta'] = value
        for key in ('tasks', 'responses_received', 'format_valid', 'transport_unresolved', 'not_attempted'):
            total = sum(r[key] for r in rows)
            assert total == target['overall'][key], (model, key)
            result['transport'][key] = total
        def gain(bg, level, pattern):
            return int(lookup[bg, level, pattern, 'U']['workflow_success']) - int(lookup[bg, level, pattern, 'D']['workflow_success'])
        contrast = np.array([gain(bg, 'high', 'staggered_slots') - gain(bg, 'high', 'synchronized_slots')
                             - gain(bg, 'low', 'staggered_slots') + gain(bg, 'low', 'synchronized_slots') for bg in bgs])
        planned = target['paired_statistics']['estimates']['predeclared_workflow_interaction']
        check_interval(interval(contrast, idx), planned)
        family = {r['background']: r['family'] for r in rows}
        result['leave_one_template_out'] = {}
        for omitted in (0, 1, 2):
            keep = [i for i, bg in enumerate(bgs) if family[bg] != omitted]
            assert len(keep) == 8
            # In this design the low-overlap contrast is zero; do not silently equate different estimands.
            assert all(gain(bg, 'low', 'staggered_slots') == gain(bg, 'low', 'synchronized_slots') == 0 for bg in bgs)
            value = float(contrast[keep].mean())
            close(value, planned['leave_one_family_out'][str(omitted)])
            result['leave_one_template_out'][str(omitted)] = value
        # Existing post-hoc complete-transport sensitivity, without replacing main denominators.
        result['complete_transport'] = {}
        for condition, wanted in target['complete_transport'].items():
            level, pattern = condition.split('/')
            paired = [(lookup[bg, level, pattern, 'D'], lookup[bg, level, pattern, 'U']) for bg in bgs]
            paired = [(d, u) for d, u in paired if d['responses_received'] == u['responses_received'] == 3]
            actual = {'backgrounds': len(paired),
                'D': sum(d['workflow_success'] for d, u in paired), 'U': sum(u['workflow_success'] for d, u in paired)}
            assert actual == {'backgrounds': wanted['complete_transport_pairs'],
                              'D': wanted['D_workflow_success'], 'U': wanted['U_workflow_success']}, (model, condition)
            result['complete_transport'][condition] = actual
        report['business'][model] = result

    # Ratio of pooled target counts, resampling paired background clusters.
    assert len(probe) == 384
    rng = np.random.default_rng(20260915)
    for mode in ('curious', 'injection'):
        rows = [r for r in probe if r['mode'] == mode]
        bgs = sorted({r['background'] for r in rows})
        assert len(bgs) == 24
        lookup = {(r['key'], r['task_index'], r['condition']): r for r in rows}
        assert len(lookup) == 192
        for r in rows:
            assert r['target_count'] == lookup[r['key'], r['task_index'], 'U' if r['condition'] == 'D' else 'D']['target_count']
        result = {}
        arrays = {}
        for arm in ('D', 'U'):
            part = [r for r in rows if r['condition'] == arm]
            assert len(part) == 96
            total = {'target_values_extracted': sum(r['leaked_target_count'] for r in part),
                     'counterfactual_target_values': sum(r['target_count'] for r in part),
                     'tasks_with_any_target_extracted': sum(r['any_target_leaked'] for r in part),
                     'business_argument_exact': sum(r['argument_exact'] for r in part),
                     'business_goal_exact': sum(r['goal_exact'] for r in part)}
            for key, value in total.items():
                assert value == expected['probe']['groups'][mode][arm][key]
            result[arm] = total
            arrays[arm] = np.array([[sum(r[k] for r in part if r['background'] == bg)
                                    for k in ('leaked_target_count', 'target_count', 'any_target_leaked', 'goal_exact')]
                                   + [sum(r['background'] == bg for r in part)] for bg in bgs], dtype=float)
        idx = rng.integers(0, 24, (10000, 24))
        d, u = arrays['D'][idx].sum(axis=1), arrays['U'][idx].sum(axis=1)
        raw_d, raw_u = arrays['D'].sum(axis=0), arrays['U'].sum(axis=0)
        for name, num, den in [('target_extraction_rate_diff', 0, 1), ('any_extraction_task_rate_diff', 2, 4), ('business_goal_rate_diff', 3, 4)]:
            actual = {'estimate': float(raw_u[num] / raw_u[den] - raw_d[num] / raw_d[den]),
                      'ci95': np.quantile(u[:, num] / u[:, den] - d[:, num] / d[:, den], [.025, .975]).tolist()}
            wanted = expected['probe']['paired'][mode][name]
            close(actual['estimate'], wanted['estimate']); close(actual['ci95'], wanted['ci95'])
            result[name] = actual
        report['probe'][mode] = result
    report['status'] = 'PASS'
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))

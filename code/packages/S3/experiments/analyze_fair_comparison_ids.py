"""Dev-only calibration and paired analysis for the fixed-contract comparison.

Every test threshold is predetermined. Operating points are locked using dev
records before test calls; test results cannot choose a replacement threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from review_revision import ROOT, OUT, read, dump, digest
from review_request_visibility import VISIBILITY_VERSION, request_fact_visibility, score_visible
from analyze_review_revision import aggregate

FAIR = OUT/'fair_comparison_ids'
MODELS = ('qwen', 'glm', 'gpt55_responses')
METHODS = ('joint', 'independent')
TARGETS = (.8, .9)
SEED = 20260910
TOLERANCE = .02
BOOTSTRAPS = 10000
ANALYSIS_VERSION = 'fixed-contract-ids-dev-calibrated-paired-1'
MODEL_LABELS = {'qwen': 'Qwen3.6-27B', 'glm': 'GLM5.2', 'gpt55_responses': 'GPT-5.5'}
FIELD_DIAGNOSTIC_VERSION = 'fixed-request-endpoint-field-partition-with-output-failures-1'
FIELD_CATEGORIES = ('source_unscorable', 'method_output_failure', 'fixed_request_endpoint_gap', 'memory_selection_loss', 'retained')


def dataset(name, base=FAIR):
    raw = read(base/'datasets'/f'{name}.json')
    rows = raw['cases'] if isinstance(raw, dict) else raw
    if len({c['key'] for c in rows}) != len(rows):
        raise ValueError('duplicate_dataset_key')
    return raw, rows


def protocol_hash(base=FAIR):
    return digest(read(base/'datasets'/'protocol.json'))


def load_records(model, rows, raw, base=FAIR):
    expected_dataset_hash = digest(raw)
    expected_protocol_hash = protocol_hash(base)
    calibration_path = base/'calibration.json'
    expected_calibration_hash = digest(read(calibration_path)) if calibration_path.exists() else None
    records = []
    for case in rows:
        path = base/model/'cases'/f"{case['key']}.json"
        if not path.exists():
            raise ValueError(f'incomplete_records:{model}:{case["key"]}')
        r = read(path)
        if not r.get('complete'):
            raise ValueError('incomplete_record:'+str(path))
        if r['input_hash'] != digest(case) or r['dataset_hash'] != expected_dataset_hash:
            raise ValueError('record_dataset_hash_mismatch:'+str(path))
        if r['protocol_hash'] != expected_protocol_hash:
            raise ValueError('record_protocol_hash_mismatch:'+str(path))
        if r.get('dataset_split') == 'test' and (expected_calibration_hash is None or r.get('calibration_hash') != expected_calibration_hash):
            raise ValueError('record_calibration_hash_mismatch:'+str(path))
        if r['fixed_tasks'] != case['fixed_tasks']:
            raise ValueError('fixed_task_contract_mutated:'+str(path))
        for method in METHODS:
            status = r['method_status'][method]['status']
            if status not in ('ok','format_failure','transport_failure'):
                raise ValueError('unknown_method_status:'+str(path))
            if (status == 'ok') != isinstance(r[f'{method}_tasks'], list):
                raise ValueError('method_status_and_grades_disagree:'+str(path))
        records.append(r)
    return records


def build_visibility(case):
    bytask, unknown = {}, {}
    for task in case['fixed_tasks']:
        i = task['task_index']
        visible, unresolved = set(), set()
        for fact in case['entities']:
            if fact['source'] != 'request':
                continue
            status = request_fact_visibility(fact, task['request_quotes'], case['item']['user_input'])['status']
            if status == 'present':
                visible.add(fact['id'])
            elif status == 'unknown':
                unresolved.add(fact['id'])
        bytask[i], unknown[i] = visible, unresolved
    return bytask, unknown


def memory_decomposition(tasks, mapping, case):
    memory = {e['id'] for e in case['entities'] if e['source'] == 'memory'}
    selected_by_task = {i: set(t['keep_entity_ids']) for t, i in zip(tasks, mapping)}
    n_needed = n_kept_needed = n_negative = n_kept_negative = n_disputed = n_kept_disputed = 0
    for task in case['fixed_tasks']:
        i = task['task_index']
        needed = memory & set(case['reference']['needed'].get(str(i), []))
        possible = memory & set(case['reference']['possibly_needed'].get(str(i), []))
        negative, disputed = memory-possible, possible-needed
        selected = selected_by_task.get(i, set())
        n_needed += len(needed)
        n_kept_needed += len(selected & needed)
        n_negative += len(negative)
        n_kept_negative += len(selected & negative)
        n_disputed += len(disputed)
        n_kept_disputed += len(selected & disputed)
    return {'memory_needed_total': n_needed, 'memory_needed_kept': n_kept_needed,
            'memory_definitely_unnecessary_total': n_negative,
            'memory_definitely_unnecessary_kept': n_kept_negative,
            'memory_disputed_total': n_disputed, 'memory_disputed_kept': n_kept_disputed,
            'memory_required_recall': n_kept_needed/n_needed if n_needed else None,
            'memory_false_positive_rate': n_kept_negative/n_negative if n_negative else None}


def field_error_diagnosis(case, selected, all_memory_endpoint):
    """Partition fields against tau=0 without changing any field's composition.

    Endpoint gaps are scored coverage gaps under the frozen fact representation;
    they do not establish a pure semantic error in request extraction. When both
    request coverage and Memory selection fail, the endpoint category takes
    precedence and the co-occurring Memory omissions are explicitly retained.
    """
    def field_map(metric):
        rows = metric['fields']
        result = {(r['task_index'], r['field']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate_diagnostic_field')
        return result
    def visible_map(metric):
        bytask = defaultdict(set)
        for row in metric['visibility']:
            bytask[row['task_index']].update(row['visible'])
        return bytask
    selected_fields, endpoint_fields = field_map(selected), field_map(all_memory_endpoint)
    reference = {(r['task_index'], r['field']): r for r in case['reference']['bindings']}
    if selected_fields.keys() != endpoint_fields.keys() or selected_fields.keys() != reference.keys():
        raise ValueError('diagnostic_field_universe_mismatch')
    selected_visible, endpoint_visible = visible_map(selected), visible_map(all_memory_endpoint)
    fact_map = {e['id']: e for e in case['entities']}
    memory = {eid for eid, fact in fact_map.items() if fact['source'] == 'memory'}
    counts = {category: 0 for category in FIELD_CATEGORIES}
    rows = []
    for key, binding in reference.items():
        ti, field = key
        chosen, endpoint = selected_fields[key], endpoint_fields[key]
        if chosen['eligible'] != endpoint['eligible']:
            raise ValueError('diagnostic_eligibility_changed')
        required = set(binding['entity_ids'])
        missing_endpoint = required-endpoint_visible[ti]
        missing_selected = required-selected_visible[ti]
        missing_memory = missing_selected & memory
        if not chosen['eligible']:
            category = 'source_unscorable'
        elif selected.get('method_output_failure'):
            if chosen['correct'] or selected_visible[ti]:
                raise ValueError('failed_method_must_not_send_payload')
            category = 'method_output_failure'
        elif not endpoint['correct']:
            if chosen['correct']:
                raise ValueError('selected_not_subset_of_all_memory_endpoint')
            if missing_endpoint & memory:
                raise ValueError('tau_zero_endpoint_missing_memory')
            category = 'fixed_request_endpoint_gap'
        elif not chosen['correct']:
            if not missing_selected or missing_selected-memory:
                raise ValueError('memory_selection_changed_request_visibility')
            category = 'memory_selection_loss'
        else:
            category = 'retained'
        if chosen['eligible'] and chosen['correct'] != (not missing_selected):
            raise ValueError('diagnostic_visibility_score_disagreement')
        counts[category] += 1
        rows.append({'task_index': ti, 'field': field, 'category': category,
            'reference_status': binding['status'], 'task_active': case['reference']['tasks'][ti]['active'],
            'required_entity_ids': sorted(required), 'missing_at_tau_zero': sorted(missing_endpoint),
            'missing_at_selected_threshold': sorted(missing_selected), 'missing_memory_ids': sorted(missing_memory),
            'cooccurring_memory_loss_in_endpoint_gap': category == 'fixed_request_endpoint_gap' and bool(missing_memory),
            'counterfactual_endpoint_gap_on_failed_method': category == 'method_output_failure' and not endpoint['correct'],
            'missing_entity_sources': {eid: fact_map[eid]['source'] for eid in sorted(missing_selected)}})
    if counts['retained'] != selected['correct'] or sum(counts.values()) != selected['total']:
        raise ValueError('diagnostic_count_conservation_failed')
    return {'key': case['key'], 'split': case['split'], 'counts': counts,
            'method_output_failure': selected.get('method_output_failure', False),
            'all_fields': selected['total'], 'eligible_fields': selected['eligible_total'],
            'tau_zero_correct': all_memory_endpoint['correct'], 'selected_correct': selected['correct'],
            'cooccurring_memory_loss_in_endpoint_gap': sum(r['cooccurring_memory_loss_in_endpoint_gap'] for r in rows),
            'fields': rows}


def summarize_field_diagnoses(rows):
    counts = {category: sum(row['counts'][category] for row in rows) for category in FIELD_CATEGORIES}
    return {'protocol': FIELD_DIAGNOSTIC_VERSION, 'n_inputs': len(rows), 'field_counts': counts,
            'method_output_failure_inputs': sum(row['method_output_failure'] for row in rows),
            'mean_fields_per_input': {category: counts[category]/len(rows) for category in FIELD_CATEGORIES},
            'inputs_with_category': {category: sum(row['counts'][category] > 0 for row in rows) for category in FIELD_CATEGORIES},
            'all_fields': sum(row['all_fields'] for row in rows),
            'eligible_fields': sum(row['eligible_fields'] for row in rows),
            'tau_zero_correct': sum(row['tau_zero_correct'] for row in rows),
            'selected_correct': sum(row['selected_correct'] for row in rows),
            'cooccurring_memory_loss_in_endpoint_gap': sum(row['cooccurring_memory_loss_in_endpoint_gap'] for row in rows),
            'source_unscorable_reasons': dict(Counter('inactive_task' if not f['task_active'] else f['reference_status']
                for row in rows for f in row['fields'] if f['category'] == 'source_unscorable'))}


def fixed_all_memory_endpoint(case):
    """A deterministic coverage ceiling for diagnosis, independent of model status."""
    visible, _ = build_visibility(case)
    memory = {e['id'] for e in case['entities'] if e['source'] == 'memory'}
    tasks = [{'name': t['name'], 'request_quotes': t['request_quotes'], 'keep_entity_ids': sorted(memory)} for t in case['fixed_tasks']]
    mapping = [t['task_index'] for t in case['fixed_tasks']]
    byidentity = {id(t): i for t,i in zip(tasks,mapping)}
    return score_visible(tasks,mapping,case['entities'],case['reference'],
                         lambda t,fs,ids: memory | visible[byidentity[id(t)]])


def evaluate_curve(rows, records):
    from review_fair_comparison_ids import materialize_tasks, thresholds
    visibility = [build_visibility(case) for case in rows]
    curves = {method: {} for method in METHODS}
    for method in METHODS:
        for threshold in thresholds():
            metrics = []
            for case, record, (visible, unknown) in zip(rows, records, visibility):
                status = record['method_status'][method]['status']
                if status != 'ok':
                    if record[f'{method}_tasks'] is not None:
                        raise ValueError('failed_method_must_not_have_imputed_grades')
                    metric = score_visible([], [], case['entities'], case['reference'], lambda *args: set())
                    metric.update(memory_decomposition([], [], case))
                    metric.update({'request_unknown_fact_recipient_pairs': 0,
                        'request_unknown_could_change_fields': 0, 'P_available_lexical_ceiling': metric['P_available'],
                        'L_all_upper_with_unknown': 0, 'key': case['key'], 'split': case['split'],
                        'method_output_failure': True, 'method_status': status})
                    metrics.append(metric)
                    continue
                tasks, mapping = materialize_tasks(case, record[f'{method}_tasks'], threshold)
                if mapping != [t['task_index'] for t in case['fixed_tasks']]:
                    raise ValueError('materialized_mapping_changed')
                byname = {id(task): gi for task, gi in zip(tasks, mapping)}
                def get_visible(task, facts, ids, include_unknown=False):
                    gi = byname[id(task)]
                    result = set(task['keep_entity_ids']) | visible[gi]
                    return result | unknown[gi] if include_unknown else result
                metric = score_visible(tasks, mapping, case['entities'], case['reference'], get_visible)
                ceiling = score_visible(tasks, mapping, case['entities'], case['reference'],
                    lambda task, facts, ids: get_visible(task, facts, ids, True))
                metric.update(memory_decomposition(tasks, mapping, case))
                metric.update({'request_unknown_fact_recipient_pairs': sum(len(v) for v in unknown.values()),
                               'request_unknown_could_change_fields': ceiling['correct']-metric['correct'],
                               'P_available_lexical_ceiling': ceiling['P_available'],
                               'L_all_upper_with_unknown': ceiling['L_all_upper'],
                               'key': case['key'], 'split': case['split'],
                               'method_output_failure': False, 'method_status': status})
                metrics.append(metric)
            curves[method][f'{threshold:.1f}'] = metrics
    return curves


def summarize_metrics(metrics):
    result = aggregate(metrics)
    if result is None:
        return None
    count_fields = ('memory_needed_total','memory_needed_kept','memory_definitely_unnecessary_total',
                    'memory_definitely_unnecessary_kept','memory_disputed_total','memory_disputed_kept',
                    'request_unknown_fact_recipient_pairs','request_unknown_could_change_fields')
    result.update({k: sum(m[k] for m in metrics) for k in count_fields})
    result['method_output_failure_inputs'] = sum(m['method_output_failure'] for m in metrics)
    result['valid_output_inputs'] = len(metrics)-result['method_output_failure_inputs']
    result['valid_output_rate'] = result['valid_output_inputs']/len(metrics)
    result['format_failure_inputs'] = sum(m['method_status'] == 'format_failure' for m in metrics)
    result['transport_failure_inputs'] = sum(m['method_status'] == 'transport_failure' for m in metrics)
    for key in ('memory_required_recall','memory_false_positive_rate','P_available_lexical_ceiling'):
        values = [m[key] for m in metrics if m[key] is not None]
        result[key+'_macro'] = float(np.mean(values)) if values else None
    result['memory_required_recall_micro'] = (result['memory_needed_kept']/result['memory_needed_total']
                                            if result['memory_needed_total'] else None)
    result['memory_false_positive_rate_micro'] = (result['memory_definitely_unnecessary_kept']/result['memory_definitely_unnecessary_total']
                                                 if result['memory_definitely_unnecessary_total'] else None)
    result['L_all_upper_with_unknown'] = float(np.mean([m['L_all_upper_with_unknown'] for m in metrics]))
    return result


def choose_threshold(points, target):
    """Only dev points may be passed. A failed target gets no fallback threshold."""
    feasible = [p for p in points if p['P_available_macro'] is not None and p['P_available_macro'] >= target]
    if not feasible:
        return {'target': target, 'reachable': False, 'threshold': None,
                'max_dev_retention': max((p['P_available_macro'] for p in points if p['P_available_macro'] is not None), default=None),
                'reason': 'target_unreachable_on_dev_no_test_threshold_selected'}
    best = min(feasible, key=lambda p: (p['L_upper'], -p['threshold']))
    return {'target': target, 'reachable': True, 'threshold': best['threshold'],
            'dev_retention': best['P_available_macro'], 'dev_L_lower': best['L_lower'],
            'dev_L_upper': best['L_upper']}


def locked_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('locked_calibration_differs_do_not_recalibrate_after_test')
    else:
        dump(path, value)


def calibrate(models=MODELS, base=FAIR):
    raw, rows = dataset('dev', base)
    if any(c['key'].startswith(('multi_','single_','neutral_')) for c in rows):
        raise ValueError('dev_contains_existing_test_keys')
    result = {'analysis_protocol': ANALYSIS_VERSION, 'visibility_protocol': VISIBILITY_VERSION,
              'selection_data': 'dev_only', 'dev_dataset_hash': digest(raw), 'protocol_hash': protocol_hash(base),
              'targets': list(TARGETS), 'selection_rule': 'P_available_macro >= target; minimum L_upper; ties prefer higher threshold',
              'test_matching_tolerance': TOLERANCE, 'test_reselection_permitted': False,
              'method_failure_policy': 'transactional_no_payload; operational P=0,L=0; never impute grades',
              'models': {}}
    for model in models:
        records = load_records(model, rows, raw, base)
        curves = evaluate_curve(rows, records)
        points = {method: [{'threshold': float(t), **summarize_metrics(ms)} for t, ms in curves[method].items()]
                  for method in METHODS}
        result['models'][model] = {'n_dev': len(rows), 'dev_record_hashes': {r['key']: digest(r) for r in records},
            'targets': {f'{target:.1f}': {method: choose_threshold(points[method], target) for method in METHODS} for target in TARGETS}}
        dump(base/'analysis'/'dev_curves'/f'{model}.json', {'model': model, 'points': points})
    locked_json(base/'calibration.json', result)
    return result


def paired_ci(joint, independent, seed=SEED, draws=BOOTSTRAPS):
    """Input-cluster bootstrap, direction joint minus independent, no field resampling."""
    if len(joint) != len(independent) or not joint:
        raise ValueError('paired_ci_requires_nonempty_matching_inputs')
    if any((a is None) != (b is None) for a,b in zip(joint, independent)):
        raise ValueError('paired_metric_denominators_disagree')
    valid = [(a,b) for a,b in zip(joint, independent) if a is not None and b is not None]
    if not valid:
        return {'n': 0, 'difference_joint_minus_independent': None, 'ci95': None}
    delta = np.asarray([a-b for a,b in valid], dtype=float)
    rng = np.random.default_rng(seed)
    resampled = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(axis=1)
    return {'n': len(delta), 'difference_joint_minus_independent': float(delta.mean()),
            'ci95': np.quantile(resampled, [.025, .975]).tolist(), 'bootstrap_draws': draws, 'seed': seed}


def paired_ratio_ci(joint, independent, numerator, denominator, seed=SEED, draws=BOOTSTRAPS):
    """Micro-score uncertainty still resamples inputs, not their individual fields."""
    if not joint or len(joint) != len(independent):
        raise ValueError('paired_ratio_requires_matching_inputs')
    jd = np.asarray([m[denominator] for m in joint], dtype=float)
    ind = np.asarray([m[denominator] for m in independent], dtype=float)
    if not np.array_equal(jd, ind):
        raise ValueError('paired_ratio_denominators_disagree')
    if not jd.sum():
        return {'n': len(joint), 'difference_joint_minus_independent': None, 'ci95': None}
    difference = np.asarray([a[numerator]-b[numerator] for a,b in zip(joint, independent)], dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(joint), (draws, len(joint)))
    denominators = jd[indexes].sum(axis=1)
    valid = denominators > 0
    samples = difference[indexes].sum(axis=1)[valid]/denominators[valid]
    return {'n': len(joint), 'difference_joint_minus_independent': float(difference.sum()/jd.sum()),
            'ci95': np.quantile(samples, [.025,.975]).tolist(), 'bootstrap_draws': draws,
            'valid_bootstrap_draws': int(valid.sum()), 'seed': seed}


def retention_matches(joint_p, independent_p):
    return abs(joint_p-independent_p) <= TOLERANCE+1e-12


def target_attainment(retention, target):
    """Descriptive absolute target check; unrelated to between-method proximity."""
    return {'development_target': target, 'test_P_available_macro': retention,
            'test_target_attained': retention >= target if retention is not None else None,
            'shortfall_percentage_points': 100*max(0., target-retention) if retention is not None else None}


def dev_selection_diagnostics(targets, points_by_method):
    """Describe an already locked choice, without selecting a replacement."""
    result = {}
    for method in METHODS:
        choices = {}
        selected_thresholds = []
        for target, methods in targets.items():
            choice = methods[method]
            if not choice['reachable']:
                choices[target] = {'reachable': False}
                continue
            threshold = choice['threshold']
            selected_thresholds.append(threshold)
            chosen = next(p for p in points_by_method[method] if abs(p['threshold']-threshold) < 1e-12)
            if abs(chosen['P_available_macro']-choice['dev_retention']) > 1e-12 or abs(chosen['L_upper']-choice['dev_L_upper']) > 1e-12:
                raise ValueError('dev_curve_and_locked_choice_disagree')
            dominating = [p for p in points_by_method[method]
                if p['P_available_macro'] is not None
                and p['P_available_macro'] >= chosen['P_available_macro']-1e-12
                and p['L_upper'] <= chosen['L_upper']+1e-12
                and (p['P_available_macro'] > chosen['P_available_macro']+1e-12 or p['L_upper'] < chosen['L_upper']-1e-12)]
            choices[target] = {'reachable': True, 'selected_threshold': threshold,
                'dev_retention': chosen['P_available_macro'], 'dev_L_upper': chosen['L_upper'],
                'dev_pareto_dominated_for_retention_and_L_upper': bool(dominating),
                'dominating_dev_points': [{key: p[key] for key in ('threshold','P_available_macro','L_lower','L_upper')} for p in dominating],
                'selection_unchanged': True}
        result[method] = {'targets': choices,
            'all_reachable_targets_share_one_threshold': len(selected_thresholds) > 1 and len(set(selected_thresholds)) == 1,
            'distinct_selected_thresholds': sorted(set(selected_thresholds))}
    return result


def matched_comparison(joint, independent):
    if [m['key'] for m in joint] != [m['key'] for m in independent]:
        raise ValueError('paired_case_order_mismatch')
    js, ins = summarize_metrics(joint), summarize_metrics(independent)
    difference = js['P_available_macro']-ins['P_available_macro']
    return {'joint': js, 'independent': ins, 'absolute_retention_gap': abs(difference),
            'retention_within_descriptive_tolerance': retention_matches(js['P_available_macro'], ins['P_available_macro']),
            'tolerance': TOLERANCE, 'statistical_equivalence_claimed': False,
            'contains_method_output_failures': bool(js['method_output_failure_inputs'] or ins['method_output_failure_inputs']),
            'exposure_comparison_requires_reliability_context': True,
            'paired': {key: paired_ci([m[key] for m in joint], [m[key] for m in independent]) for key in
                ('P_available','L_lower','L_upper','L_all_lower','L_all_upper','memory_required_recall','memory_false_positive_rate')},
            'paired_micro': {label: paired_ratio_ci(joint, independent, numerator, denominator)
                for label,numerator,denominator in (
                    ('P_available','correct','eligible_total'), ('P_all','correct','total'),
                    ('memory_required_recall','memory_needed_kept','memory_needed_total'),
                    ('memory_false_positive_rate','memory_definitely_unnecessary_kept','memory_definitely_unnecessary_total'))}}


def summarize_costs(records):
    keys = ('logical_requests','unique_responses','prompt_tokens','completion_tokens',
            'response_wait_seconds_sum','schema_repairs')
    return {method: {'n': len(records),
        'totals': {key: sum(r['method_costs'][method][key] for r in records) for key in keys},
        'means': {key: float(np.mean([r['method_costs'][method][key] for r in records])) for key in keys}}
        for method in METHODS}


def conditional_valid_costs(records):
    result = {}
    for method in METHODS:
        valid = [r for r in records if r['method_status'][method]['status'] == 'ok']
        result[method] = summarize_costs(valid)[method] if valid else None
    return result


def response_history_costs(model, test_records, base):
    """Audit saved-response history separately; unknown billing is never zero."""
    dev_raw, dev_rows = dataset('dev', base)
    dev_records = load_records(model, dev_rows, dev_raw, base)
    def successful_hashes(records):
        return {row['request_hash'] for record in records for row in record['call_trace'] if row.get('response_saved', True)}
    test_hashes, dev_hashes = successful_hashes(test_records), successful_hashes(dev_records)
    all_records = test_records+dev_records
    failed_method_hashes = {trace['request_hash'] for record in all_records for trace in record['call_trace']
        if record['method_status'][trace['method']]['status'] != 'ok' and trace.get('response_saved', True)}
    groups = {name: [] for name in ('test_terminal_paths','dev_terminal_paths','outside_terminal_paths')}
    failed_method_responses = []
    for path in (base/model/'calls').glob('*.json'):
        group = 'test_terminal_paths' if path.stem in test_hashes else 'dev_terminal_paths' if path.stem in dev_hashes else 'outside_terminal_paths'
        response = read(path)
        groups[group].append(response)
        if path.stem in failed_method_hashes:
            failed_method_responses.append(response)
    def usage_stats(records):
        known = [r.get('usage', {}).get('cost') for r in records if isinstance(r.get('usage', {}).get('cost'), (int,float))]
        return {'unique_saved_responses': len(records),
            'prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', r.get('usage', {}).get('input_tokens', 0)) or 0 for r in records),
            'completion_tokens': sum(r.get('usage', {}).get('completion_tokens', r.get('usage', {}).get('output_tokens', 0)) or 0 for r in records),
            'responses_with_reported_cost': len(known), 'responses_without_reported_cost': len(records)-len(known),
            'provider_reported_cost_usd_sum': sum(known) if known else None,
            'currency_note': 'Native OpenRouter cost metadata in USD; no estimated RMB conversion or missing-charge imputation.'}
    failures = [read(p) for p in (base/model/'case_failures').glob('*.json')]
    failures = [r for r in failures if not r.get('offline')]
    transport_path = base/model/'failures.jsonl'
    transport = [json.loads(line) for line in transport_path.read_text(encoding='utf-8').splitlines() if line.strip()] if transport_path.exists() else []
    stopped_array_responses = [read(p) for p in (OUT/'fair_comparison'/model/'calls').glob('*.json')]
    return {'groups': {group: usage_stats(records) for group, records in groups.items()},
            'failed_method_paths_subset': usage_stats(failed_method_responses),
            'terminal_method_failure_events': sum(record['method_status'][method]['status'] != 'ok' for record in all_records for method in METHODS),
            'terminal_method_failure_types': dict(Counter(record['method_status'][method]['status'] for record in all_records for method in METHODS
                                                        if record['method_status'][method]['status'] != 'ok')),
            'stopped_array_protocol_saved_responses_separate': usage_stats(stopped_array_responses),
            'case_failure_events': len(failures),
            'failed_trace_rows_without_saved_response': sum(not t.get('response_saved', True) for r in failures for t in r['call_trace']),
            'transport_failure_events': len(transport),
            'case_failure_exception_counts': dict(Counter(r['exception_type'] for r in failures)),
            'notes': [
                'Terminal-path costs include valid and failed method outcomes, all saved schema-repair responses, and count unique saved responses.',
                'Outside-terminal-path responses are deduplicated by request hash; successful responses reused on retry are not charged twice in this audit.',
                'Transport failures may overlap case failures; these event counts must not be added as distinct billable requests.',
                'Failed-method-path usage is an explicitly labelled subset of current terminal paths, not extra usage to add again.',
                'Stopped-array-protocol usage belongs to the separate abandoned protocol; it is never included as performance data or new-format generation.',
                'Unreturned tokens or fees are unknown, not free. Provider-reported amounts are incomplete billing evidence; reference_usd estimates are not used.'
            ]}


def verify_calibration(calibration, models, base):
    raw, rows = dataset('dev', base)
    if calibration['dev_dataset_hash'] != digest(raw) or calibration['protocol_hash'] != protocol_hash(base):
        raise ValueError('calibration_input_or_protocol_changed')
    if calibration.get('test_reselection_permitted') is not False:
        raise ValueError('test_reselection_not_prohibited')
    if calibration.get('visibility_protocol') != VISIBILITY_VERSION or calibration.get('analysis_protocol') != ANALYSIS_VERSION:
        raise ValueError('calibration_scoring_protocol_changed')
    for model in models:
        records = load_records(model, rows, raw, base)
        hashes = {r['key']: digest(r) for r in records}
        if hashes != calibration['models'][model]['dev_record_hashes']:
            raise ValueError('dev_records_changed_after_calibration')


def analyze(models=MODELS, base=FAIR):
    calibration = read(base/'calibration.json')
    verify_calibration(calibration, models, base)
    raw, rows = dataset('test', base)
    result = {'analysis_protocol': ANALYSIS_VERSION, 'visibility_protocol': VISIBILITY_VERSION,
              'complete': tuple(models) == MODELS and len(rows) == 350,
              'test_dataset_hash': digest(raw), 'calibration_hash': digest(calibration),
              'n_test': len(rows), 'test_reselection_permitted': False,
              'bootstrap_unit': 'input; same input retained as a pair', 'bootstrap_draws': BOOTSTRAPS,
              'preceding_array_protocol': 'stopped during partial test on format-only feedback; retained separately in fair_comparison',
              'method_failure_policy': 'transactional_no_payload; operational P=0,L=0; never impute grades',
              'models': {}, 'notes': [
                  'Matching is descriptive |P_joint-P_independent| <= 0.02, not a statistical equivalence test.',
                  'No test point is selected to replace a dev operating point; all predeclared curve points and endpoints are retained.',
                  'Memory necessity decomposition uses reference intersection as definite positive and complement of union as definite negative.',
                  'Unknown request visibility is reported separately and is never promoted to a semantic positive.',
                  'Field endpoint diagnostics measure scorable coverage under the fixed fact representation, not pure semantic request-extraction error.',
                  'All intervals are pointwise and unadjusted for multiple comparisons; they do not establish simultaneous confidence or familywise significance.',
                  'Precision of estimates is conditional on frozen source labels, field contracts, calibration, hash seed and a single model judgment; their uncertainty is not included.',
                  'Zero exposure caused by missing method output is a delivery failure, not evidence of better privacy selection.',
                  'Both-valid comparisons are paired conditional analyses on the joint success intersection; they do not replace the all-input operational denominator.',
                  'Saved response waits are not measured end-to-end or parallel deployment latency.',
                  'The existing 350-item test set has been inspected in earlier research; this is a frozen confirmatory comparison, not an unseen benchmark.'
              ]}
    flat_rows = []
    for model in models:
        records = load_records(model, rows, raw, base)
        curves = evaluate_curve(rows, records)
        endpoints = [fixed_all_memory_endpoint(case) for case in rows]
        diagnostics = {}
        for target, methods in calibration['models'][model]['targets'].items():
            diagnostics[target] = {}
            for method in METHODS:
                choice = methods[method]
                if choice['reachable']:
                    selected_metrics = curves[method][f"{choice['threshold']:.1f}"]
                    diagnostics[target][method] = [field_error_diagnosis(case, selected, endpoint)
                        for case, selected, endpoint in zip(rows, selected_metrics, endpoints)]
        mr = {'test_record_hashes': {r['key']: digest(r) for r in records}, 'splits': {},
              'response_history': response_history_costs(model, records, base),
              'dev_operating_point_diagnostics': dev_selection_diagnostics(calibration['models'][model]['targets'],
                  read(base/'analysis'/'dev_curves'/f'{model}.json')['points'])}
        for split in ('all', 'multi', 'single', 'neutral'):
            indexes = [i for i,c in enumerate(rows) if split == 'all' or c['split'] == split]
            if not indexes:
                continue
            split_curves = {method: {t: [ms[i] for i in indexes] for t,ms in curve.items()}
                            for method,curve in curves.items()}
            both_valid_indexes = [i for i in indexes if all(records[i]['method_status'][method]['status'] == 'ok' for method in METHODS)]
            sr = {'n': len(indexes), 'curves': {}, 'both_valid_curves': {}, 'both_valid_n': len(both_valid_indexes), 'targets': {},
                  'costs': summarize_costs([records[i] for i in indexes]),
                  'valid_output_costs': conditional_valid_costs([records[i] for i in indexes])}
            for method in METHODS:
                sr['curves'][method] = []
                sr['both_valid_curves'][method] = []
                for t, metrics in split_curves[method].items():
                    point = {'threshold': float(t), **summarize_metrics(metrics)}
                    sr['curves'][method].append(point)
                    flat_rows.append({'model': model, 'split': split, 'method': method, 'population': 'all_inputs_operational', **point})
                    if both_valid_indexes:
                        conditional = {'threshold': float(t), **summarize_metrics([curves[method][t][i] for i in both_valid_indexes])}
                        sr['both_valid_curves'][method].append(conditional)
                        flat_rows.append({'model': model, 'split': split, 'method': method, 'population': 'both_valid_conditional', **conditional})
            for target, methods in calibration['models'][model]['targets'].items():
                selection = {'calibration': methods, 'comparison_available': all(x['reachable'] for x in methods.values())}
                selection['test_target_attainment'] = {method: target_attainment(
                    summarize_metrics(split_curves[method][f"{methods[method]['threshold']:.1f}"])['P_available_macro'] if methods[method]['reachable'] else None,
                    float(target)) for method in METHODS}
                selection['field_error_diagnosis'] = {method: summarize_field_diagnoses([diagnostics[target][method][i] for i in indexes])
                    for method in METHODS if method in diagnostics[target]}
                if selection['comparison_available']:
                    selected = {method: split_curves[method][f"{methods[method]['threshold']:.1f}"] for method in METHODS}
                    selection.update(matched_comparison(selected['joint'], selected['independent']))
                    both_indexes = [i for i,(j,ind) in enumerate(zip(selected['joint'], selected['independent']))
                                    if not j['method_output_failure'] and not ind['method_output_failure']]
                    selection['both_valid'] = {'n_inputs': len(both_indexes), 'fraction_of_all_inputs': len(both_indexes)/len(indexes),
                        'comparison_available': bool(both_indexes), 'selection': 'both methods have valid outputs; paired conditional analysis only'}
                    if both_indexes:
                        selection['both_valid'].update(matched_comparison(
                            [selected['joint'][i] for i in both_indexes], [selected['independent'][i] for i in both_indexes]))
                        selection['both_valid']['test_target_attainment'] = {method: target_attainment(
                            selection['both_valid'][method]['P_available_macro'], float(target)) for method in METHODS}
                    selection['both_valid']['keys'] = [selected['joint'][i]['key'] for i in both_indexes]
                else:
                    selection['reason'] = 'one_or_both_dev_targets_unreachable; no test fallback'
                sr['targets'][target] = selection
            mr['splits'][split] = sr
        result['models'][model] = mr
        # Per-input curve metrics permit independent reproduction of every paired comparison.
        dump(base/'analysis'/'per_input'/f'{model}.json', {'model': model, 'curves': curves})
        dump(base/'analysis'/'field_diagnostics'/f'{model}.json', {'model': model, 'protocol': FIELD_DIAGNOSTIC_VERSION,
            'categories': list(FIELD_CATEGORIES), 'targets': diagnostics,
            'note': 'Original-unscorable fields remain separate; method-output failures form a distinct category before the valid-output bottleneck partition. Counterfactual all-Memory endpoints never create method grades or delivered payloads.'})
        print(json.dumps({'model': model, 'test_cases_analyzed': len(rows)}, ensure_ascii=False), flush=True)
    output = base/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    dump(output/'summary.json', result)
    with (output/'test_curves.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    plot_curves(result, output)
    write_report(result, output)
    return result


def plot_curves(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['DengXian','DejaVu Sans'], 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'font.size': 10})
    for population, curve_key, population_label in (
        ('operational','curves','全部输入；失败不发送载荷'),
        ('both_valid','both_valid_curves','双方合法输出交集；条件分析')):
        for split in ('all','multi','single','neutral'):
            present = [(m,r['splits'][split]) for m,r in result['models'].items() if split in r['splits']]
            if not present:
                continue
            fig, axes = plt.subplots(1, len(present), figsize=(5*len(present), 4), squeeze=False)
            for ax, (model, sr) in zip(axes[0], present):
                for method, label, color in [('joint','联合判断','#2563a8'),('independent','逐任务判断','#c36b20')]:
                    points = sr[curve_key][method]
                    if not points:
                        continue
                    x = [100*p['P_available_macro'] for p in points]
                    low, high = [p['L_lower'] for p in points], [p['L_upper'] for p in points]
                    ax.plot(x, high, label=label, color=color, linewidth=1.8)
                    ax.fill_between(x, low, high, color=color, alpha=.14)
                    for target, selected in sr['targets'].items():
                        choice = selected['calibration'][method]
                        if not choice['reachable']:
                            continue
                        point = next(p for p in points if abs(p['threshold']-choice['threshold']) < 1e-8)
                        ax.scatter(100*point['P_available_macro'], point['L_upper'], marker='o' if target == '0.8' else 's', color=color, s=42)
                n = sr['n'] if population == 'operational' else sr['both_valid_n']
                ax.set(title=f'{MODEL_LABELS[model]} (n={n})', xlabel='必要字段保留率（%）', ylabel='额外 Memory 披露 / 输入')
                ax.grid(alpha=.18)
                if n:
                    ax.legend(frameon=False)
            fig.suptitle(f'{split}：{population_label}\n全部预设阈值；阴影为必要性分歧区间，标记为开发集锁定点')
            fig.tight_layout()
            fig.savefig(output/f'curve_{split}_{population}_ids.png', dpi=180)
            fig.savefig(output/f'curve_{split}_{population}_ids.pdf')
            plt.close(fig)


def fmt(value, percent=False):
    if value is None:
        return '不可用'
    return f'{100*value:.2f}%' if percent else f'{value:.2f}'


def attainment_pair(point):
    labels = []
    for method in METHODS:
        value = point['test_target_attainment'][method]['test_target_attained']
        labels.append('不可用' if value is None else '是' if value else '否')
    return ' / '.join(labels)


def write_report(result, output):
    status = '完整' if result['complete'] else '部分模型结果；不能标为完整'
    lines = ['# 固定任务合同的联合与逐任务比较', '', f'状态：{status}。每模型测试 {result["n_test"]} 条输入。', '',
        '先行数组格式协议在部分测试期间因任务×事实数组覆盖格式失败而中止；变更依据是语法校验反馈，不是测试语义得分。旧协议及调用原样留在 `fair_comparison/`。',
        '本报告来自 `fair_comparison_ids/` 的新协议：所有模型、两种方法重新运行全部开发集并校准，再运行全部测试集。模型输出显式事实 ID→等级，不复用旧数组格式判断。',
        '任务、请求片段、全部候选 Memory 事实及字段合同对两种方法完全相同。模型只输出各事实的必要性等级。',
        '预先采用事务失败不发送策略：方法内任一必要路径在规定预算内无合法输出，则整个方法不发送 Memory 或固定请求；字段分母保留，操作性 P=0、L=0，不补造等级。格式或传输失败单列，失败产生的零披露不视为隐私优势。',
        '对所有预设阈值进行确定性保留，整数阈值对应纯等级切分，小数阈值按与方法无关的固定哈希平滑。',
        '开发集分别以 80% 和 90% 保留目标，选择达到目标且 Memory 披露上界最小的阈值；平局取更高阈值。测试集没有重新选点。',
        '“测试达到开发目标”逐方法检查测试保留率是否达到该行的 80% 或 90% 绝对目标。“方法间相近”仅检查两方法保留率差是否不超过 2 个百分点，是描述性容差，既不代表统计等效，也不意味着达到开发目标。两个低保留率方法可能彼此相近但均不达标。',
        '测试集为已经研究过的冻结 350 条数据，开发集为新增 42 条模板数据。开发集全部参考字段放在 Memory 中，且文风和难度与测试集不同；校准转移可能失败，不能把测试结果称为全新未见样本的泛化验证。', '',
        '| 模型 | 划分 | dev目标 | 联合τ / 逐任务τ | 联合P / 逐任务P | 测试达开发目标（联/逐） | 方法间相近 | 联合L / 逐任务L（上界） | ΔL 95% CI（联合−逐任务） |',
        '|---|---|---:|---:|---:|---|---|---:|---|']
    tex_rows = []
    conditional_tex_rows = []
    for model, mr in result['models'].items():
        for split, sr in mr['splits'].items():
            for target, point in sr['targets'].items():
                cal = point['calibration']
                if not point['comparison_available']:
                    lines.append(f'| {MODEL_LABELS[model]} | {split} | {float(target):.0%} | 不可达 | — | 不可用 | 不比较 | — | — |')
                    continue
                joint, independent = point['joint'], point['independent']
                match = '是' if point['retention_within_descriptive_tolerance'] else '否，保留差超容差'
                attained = attainment_pair(point)
                ci = point['paired']['L_upper']['ci95']
                lines.append(f"| {MODEL_LABELS[model]} | {split} | {float(target):.0%} | {cal['joint']['threshold']:.1f} / {cal['independent']['threshold']:.1f} | {fmt(joint['P_available_macro'],True)} / {fmt(independent['P_available_macro'],True)} | {attained} | {match} | {joint['L_upper']:.2f} / {independent['L_upper']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
                if split == 'multi':
                    tex_rows.append(f"{MODEL_LABELS[model]} & {100*float(target):.0f}\\% & {100*joint['P_available_macro']:.2f} / {100*independent['P_available_macro']:.2f} & {joint['L_upper']:.2f} / {independent['L_upper']:.2f} & {attained} & {'是' if point['retention_within_descriptive_tolerance'] else '否'} & [{ci[0]:.2f}, {ci[1]:.2f}]" + r' \\')
    lines += ['', '开发集锁定点的描述性检查（不更换已冻结阈值）：', '',
        '| 模型 | 方法 | 开发目标 | 锁定τ | 锁定点dev P | 是否被其他dev点支配 | 较近的支配点（τ/P/L上界） |',
        '|---|---|---|---:|---:|---|---|']
    same_target_points = True
    for model,mr in result['models'].items():
        for method,diagnostic in mr['dev_operating_point_diagnostics'].items():
            same_target_points &= diagnostic['all_reachable_targets_share_one_threshold']
            seen = set()
            for target,choice in diagnostic['targets'].items():
                if not choice['reachable'] or choice['selected_threshold'] in seen:
                    continue
                threshold = choice['selected_threshold']
                seen.add(threshold)
                target_labels = '/'.join(f'{100*float(t):.0f}%' for t,c in diagnostic['targets'].items()
                    if c['reachable'] and c['selected_threshold'] == threshold)
                dominators = choice['dominating_dev_points']
                nearest = min(dominators,key=lambda p:abs(p['threshold']-threshold)) if dominators else None
                evidence = f"{nearest['threshold']:.1f} / {100*nearest['P_available_macro']:.3f}% / {nearest['L_upper']:.3f}" if nearest else '无'
                lines.append(f"| {MODEL_LABELS[model]} | {'联合' if method == 'joint' else '逐任务'} | {target_labels} | {threshold:.1f} | {100*choice['dev_retention']:.3f}% | {'是' if dominators else '否'} | {evidence} |")
    if same_target_points:
        lines += ['', '所有模型／方法的两个开发目标落在同一阈值点；80% 与 90% 两行不是两次独立验证。']
    dev_choices = [choice for mr in result['models'].values() for diagnostic in mr['dev_operating_point_diagnostics'].values()
                   for choice in diagnostic['targets'].values() if choice['reachable']]
    dominated_31 = bool(dev_choices) and all(abs(choice['selected_threshold']-3.1) < 1e-12
        and any(abs(p['threshold']-3.) < 1e-12 and abs(p['L_upper']-choice['dev_L_upper']) < 1e-12
                and p['P_available_macro'] > choice['dev_retention'] for p in choice['dominating_dev_points']) for choice in dev_choices)
    if dominated_31:
        lines += ['本次预设的“披露上界平局取更高阈值”规则选择了 τ=3.1。开发集 τ=3.0 的披露上界相同而保留率更高，因此所选点不是该开发集保留率—披露上界的 Pareto 最优点；小数阈值在此引入了没有披露改善的确定性哈希丢失。这是选择规则的局限，结果保留原冻结阈值，不用测试得分重新选择。']
    lines += ['', '方法可靠性（所有测试输入保留在分母中；共同成功集不替代主分析）：', '',
        '| 模型 | 划分 | 方法 | 合法输出 / 全部 | 格式失败 | 传输失败 | 双方共同合法输入 |',
        '|---|---|---|---:|---:|---:|---:|']
    for model,mr in result['models'].items():
        for split,sr in mr['splits'].items():
            for method in METHODS:
                point = sr['curves'][method][0]
                lines.append(f"| {MODEL_LABELS[model]} | {split} | {'联合' if method == 'joint' else '逐任务'} | {point['valid_output_inputs']} / {sr['n']} | {point['format_failure_inputs']} | {point['transport_failure_inputs']} | {sr['both_valid_n']} |")
    lines += ['', '双方均合法输出的配对交集（条件分析，继续使用原开发集阈值，不重新选点）：', '',
        '| 模型 | 划分 | dev目标 | 共同合法 n | 联合P / 逐任务P | 测试达开发目标（联/逐） | 方法间相近 | 联合L / 逐任务L（上界） | ΔL 95% CI |',
        '|---|---|---:|---:|---:|---|---|---:|---|']
    for model,mr in result['models'].items():
        for split,sr in mr['splits'].items():
            for target,operational in sr['targets'].items():
                point = operational.get('both_valid')
                if not point or not point['comparison_available']:
                    lines.append(f'| {MODEL_LABELS[model]} | {split} | {100*float(target):.0f}% | 0或开发目标不可达 | — | 不可用 | 不比较 | — | — |')
                    continue
                joint,independent = point['joint'],point['independent']
                ci = point['paired']['L_upper']['ci95']
                match = '是' if point['retention_within_descriptive_tolerance'] else '否'
                attained = attainment_pair(point)
                lines.append(f"| {MODEL_LABELS[model]} | {split} | {100*float(target):.0f}% | {point['n_inputs']} | {fmt(joint['P_available_macro'],True)} / {fmt(independent['P_available_macro'],True)} | {attained} | {match} | {joint['L_upper']:.2f} / {independent['L_upper']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
                if split == 'multi':
                    conditional_tex_rows.append(f"{MODEL_LABELS[model]} & {100*float(target):.0f}\\% & {point['n_inputs']} & {100*joint['P_available_macro']:.2f} / {100*independent['P_available_macro']:.2f} & {joint['L_upper']:.2f} / {independent['L_upper']:.2f} & {attained} & {match}" + r' \\')
    lines += ['', '字段错误来源诊断（多任务输入，各字段的全部组成事实仍须同时保留）：', '',
        '| 模型 | dev目标 | 方法 | 原始来源／参考不可评分 | 方法输出失败 | 固定请求端点覆盖缺口 | Memory选择新增丢失 | 成功保留 | 端点缺口且另有Memory遗漏 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for model, mr in result['models'].items():
        for target, point in mr['splits'].get('multi', {}).get('targets', {}).items():
            for method, diagnostic in point['field_error_diagnosis'].items():
                c = diagnostic['field_counts']
                lines.append(f"| {MODEL_LABELS[model]} | {100*float(target):.0f}% | {'联合' if method == 'joint' else '逐任务'} | {c['source_unscorable']} | {c['method_output_failure']} | {c['fixed_request_endpoint_gap']} | {c['memory_selection_loss']} | {c['retained']} | {diagnostic['cooccurring_memory_loss_in_endpoint_gap']} |")
    lines += ['', '端点为 τ=0，即保留全部 Memory、保持固定请求不变。端点仍未满足的字段表示相对于冻结事实表示的可评分覆盖缺口，不能据此断言纯语义上的请求截取错误。',
        '五类互斥且覆盖完整字段全集：先剥离不可评分字段，再单列方法输出失败；对合法输出继续识别端点缺口、Memory 选择新增丢失和保留成功。端点缺口同时伴有的 Memory 遗漏另列，不能误读为单一原因。失败方法的全 Memory 端点仅为诊断反事实，不是发送过的载荷。',
        '逐输入的各类数量、字段全部参考组成、具体 missing IDs 及其 source 保存于 `analysis/field_diagnostics/{model}.json`。', '',
        '所有输入的实际路径开销（包括失败方法已消耗的调用与格式修复；按输入统计，所有阈值共用同一次判断）：', '',
        '| 模型 | 划分 | 方法 | 总响应数 | 响应数/输入 | 输入token/输入 | 输出token/输入 | 格式修复响应数 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    cost_tex_rows = []
    for model, mr in result['models'].items():
        for split in ('all','multi'):
            if split not in mr['splits']:
                continue
            for method, cost in mr['splits'][split]['costs'].items():
                means, totals = cost['means'], cost['totals']
                label = '联合' if method == 'joint' else '逐任务'
                lines.append(f"| {MODEL_LABELS[model]} | {split} | {label} | {totals['unique_responses']} | {means['unique_responses']:.2f} | {means['prompt_tokens']:.1f} | {means['completion_tokens']:.1f} | {totals['schema_repairs']} |")
                cost_tex_rows.append(f"{MODEL_LABELS[model]} & {'全部' if split == 'all' else '多任务'} & {label} & {means['unique_responses']:.2f} & {means['prompt_tokens']:.1f} & {means['completion_tokens']:.1f} & {totals['schema_repairs']}" + r' \\')
    lines += ['', '仅合法输出方法的成功路径开销（这是按方法各自成功输入的成本描述，不作为配对效用比较）：', '',
        '| 模型 | 划分 | 方法 | 成功输入数 | 响应数/成功输入 | 输入token/成功输入 | 输出token/成功输入 |',
        '|---|---|---|---:|---:|---:|---:|']
    for model,mr in result['models'].items():
        for split in ('all','multi'):
            if split not in mr['splits']:
                continue
            for method,cost in mr['splits'][split]['valid_output_costs'].items():
                if cost is None:
                    lines.append(f"| {MODEL_LABELS[model]} | {split} | {method} | 0 | 不可用 | 不可用 | 不可用 |")
                else:
                    means = cost['means']
                    lines.append(f"| {MODEL_LABELS[model]} | {split} | {'联合' if method == 'joint' else '逐任务'} | {cost['n']} | {means['unique_responses']:.2f} | {means['prompt_tokens']:.1f} | {means['completion_tokens']:.1f} |")
    lines += ['', '开发集、测试终止路径与路径外历史响应分开记账（终止路径包含合法和失败方法）：', '',
        '| 模型 | 范围 | 保存响应数 | 已报告输入token | 已报告输出token | 有费用记录/缺失费用记录 |',
        '|---|---|---:|---:|---:|---:|']
    history_labels = {'test_terminal_paths':'测试终止路径','dev_terminal_paths':'开发校准终止路径','outside_terminal_paths':'路径外历史响应'}
    failure_notes = []
    for model, mr in result['models'].items():
        history = mr['response_history']
        for group, usage in history['groups'].items():
            lines.append(f"| {MODEL_LABELS[model]} | {history_labels[group]} | {usage['unique_saved_responses']} | {usage['prompt_tokens']} | {usage['completion_tokens']} | {usage['responses_with_reported_cost']} / {usage['responses_without_reported_cost']} |")
        for key,label in [('failed_method_paths_subset','其中：事务失败方法路径（子集，不重复加总）'),
                          ('stopped_array_protocol_saved_responses_separate','另列：已中止数组协议的全部保存响应')]:
            usage = history[key]
            lines.append(f"| {MODEL_LABELS[model]} | {label} | {usage['unique_saved_responses']} | {usage['prompt_tokens']} | {usage['completion_tokens']} | {usage['responses_with_reported_cost']} / {usage['responses_without_reported_cost']} |")
        failure_notes.append(f"{MODEL_LABELS[model]}：已终止方法失败事件 {history['terminal_method_failure_events']} 次；案例异常事件 {history['case_failure_events']} 次，其中失败trace未保存响应 {history['failed_trace_rows_without_saved_response']} 条；底层传输失败事件 {history['transport_failure_events']} 次。不同事件层级可能重叠，不能相加当作收费请求数。")
    lines += ['', *failure_notes]
    lines += ['', '原币费用仅按接口已返回记录保存于 JSON；未返回的费用和 token 不作零值计费推断。路径外历史响应单独列出，重试时复用同一已保存响应不会重复计为新响应；不使用旧价格估计替代账单。']
    lines += ['', '完整曲线及端点见 [test_curves.csv](../../experiments/results/review_revision_20260910/fair_comparison_ids/analysis/test_curves.csv)。',
        '数值全集、Memory 必要项召回、误保留与分歧项分解、请求未知数、成本及配对区间见 [summary.json](../../experiments/results/review_revision_20260910/fair_comparison_ids/analysis/summary.json)。',
        '', '区间按输入进行 10000 次配对 bootstrap，方向为联合减逐任务。Memory 的确定必要项取双标注交集，确定不必要项取双标注并集补集，分歧项单列。',
        '请求事实未知项只报告数量与词面候选上界，不补造语义真值。开销包括实际成功路径修复；等待时间总和不等于并发部署端到端延迟。',
        '这些区间覆盖输入采样差异，不覆盖模型输出随机性；每次判断只运行一次。',
        '区间精度以冻结来源标注、任务合同、开发集阈值及固定哈希种子为条件，未传播这些来源的不确定性；多个 95% 区间均为逐点区间且未作多重比较校正，不作为整体同时覆盖或家族显著性的证明。']
    revision = OUT/'request_visibility_revision/summary.json'
    if revision.exists():
        repaired = read(revision)
        lines += ['', '既有缓存评分修复：', '',
            f"已对 {repaired['cases_regraded']} 个模型×输入记录重新评分，{repaired['changed_field_condition_rows']} 个字段×条件由错改对，{repaired['unknown_field_condition_rows']} 个字段×条件仍仅有未裁决词面候选。Memory 披露不变；旧原始输出和人工材料没有修改。",
            '这属于带源位置的保守修复，不代表通用语义评分或人工验收已经完成。详见 [请求出现性审计](../../experiments/results/review_revision_20260910/request_visibility_revision/README.md)。']
    report = ROOT/'docs/paper/fair-comparison-results-20260910.md'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    tex = ['% Generated by analyze_fair_comparison_ids.py; does not overwrite historical tables.',
           '\\begin{table*}[t]', '\\centering\\small',
           '\\caption{固定任务与请求的联合／逐任务比较（多任务，全部输入）。数值及达标依次为联合／逐任务。“达标”指测试保留率达到该行开发目标；“相近”仅指两方法差不超过 2 个百分点，两者独立。失败不发送的零披露不代表隐私优势。两个目标共享同一锁定点，不是两次独立验证。}',
           '\\label{tab:fair-comparison}', '\\begin{tabular}{lrrrrrr}', '\\toprule',
           '模型 & 开发目标 & 保留率（\\%） & 披露 / 输入 & 测试达标 & 相近 & 披露差 95\\% CI \\\\', '\\midrule',
           *tex_rows, '\\bottomrule', '\\end{tabular}', '\\end{table*}']
    (ROOT/'paper-cn/fair_comparison_results_ids.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    conditional_tex = ['% Generated by analyze_fair_comparison_ids.py.', '\\begin{table*}[t]', '\\centering\\small',
        '\\caption{双方合法输出交集的条件分析（多任务）。保持开发集阈值，不重新选点；本表不替代全部输入结果。数值及是否达到开发目标依次为联合／逐任务；方法间相近不表示达到目标。两个目标共享同一锁定点。}',
        '\\label{tab:fair-comparison-both-valid}', '\\begin{tabular}{lrrrrrr}', '\\toprule',
        '模型 & 开发目标 & 共同合法 n & 保留率（\\%） & 披露 / 输入 & 测试达标 & 相近 \\\\', '\\midrule',
        *conditional_tex_rows, '\\bottomrule', '\\end{tabular}', '\\end{table*}']
    (ROOT/'paper-cn/fair_comparison_conditional_ids.tex').write_text('\n'.join(conditional_tex)+'\n', encoding='utf-8')
    cost_tex = ['% Generated by analyze_fair_comparison_ids.py.', '\\begin{table*}[t]', '\\centering\\small',
        '\\caption{固定任务与请求比较的全部输入实际路径开销，包含失败方法和修复调用。响应数、输入及输出 token 均按输入平均；格式修复列为保存的修复响应总数。阈值渲染不重新调用模型。}',
        '\\label{tab:fair-comparison-costs}', '\\begin{tabular}{lllrrrr}', '\\toprule',
        '模型 & 划分 & 方法 & 响应 / 输入 & 输入 token & 输出 token & 格式修复 \\\\', '\\midrule',
        *cost_tex_rows, '\\bottomrule', '\\end{tabular}', '\\end{table*}']
    (ROOT/'paper-cn/fair_comparison_costs_ids.tex').write_text('\n'.join(cost_tex)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('calibrate','analyze'))
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    args = parser.parse_args()
    result = calibrate(tuple(args.models)) if args.stage == 'calibrate' else analyze(tuple(args.models))
    print(json.dumps({'stage': args.stage, 'models': list(result['models']),
                      'complete': result.get('complete'), 'test_reselection_permitted': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()

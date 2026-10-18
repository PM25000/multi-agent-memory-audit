"""补齐固定合同测试的传输失败；历史实验只读，成功响应不重新采样。"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from analyze_fair_comparison_ids import FAIR, OUT, dataset, load_records, read, dump, digest
from review_fair_comparison_ids import FrozenInputs, TrackedCalls, check_calibration, run_case

RETRY = OUT / 'contract_transport_retry_20260912'
MODEL = 'gpt55_responses'


def load_completed_records(model, rows, raw):
    records = load_records(model, rows, raw)
    if model != MODEL:
        return records
    manifest = read(RETRY / 'manifest.json')
    assert manifest['complete'], 'transport_retry_incomplete'
    for index, old in enumerate(records):
        if old['key'] not in manifest['recovered_keys']:
            continue
        assert digest(old) == manifest['original_record_hashes'][old['key']]
        new = read(RETRY / MODEL / 'cases' / (old['key'] + '.json'))
        assert digest(new) == manifest['recovered_record_hashes'][old['key']]
        for field in ('input_hash', 'dataset_hash', 'protocol_hash', 'calibration_hash', 'fixed_tasks'):
            assert new[field] == old[field]
        for method in ('joint', 'independent'):
            assert new['method_status'][method]['status'] == 'ok'
            before = old['method_outcomes'][method]
            after = new['method_outcomes'][method]
            for a, b in zip(before if isinstance(before, list) else [before],
                            after if isinstance(after, list) else [after]):
                if a['status'] == 'ok':
                    assert a == b, 'successful_judgment_changed'
        records[index] = new
    return records


def run(workers=4, offline=False):
    raw, cases = dataset('test')
    originals = load_records(MODEL, cases, raw)
    failed = {r['key']: r for r in originals
              if any(s['status'] == 'transport_failure' for s in r['method_status'].values())}
    source = FAIR / MODEL
    protected = [p for directory in ('cases', 'calls', 'requests', 'request_failures')
                 for p in (source / directory).glob('*.json')]
    protected += [FAIR / 'calibration.json', FAIR / 'datasets' / 'protocol.json',
                  FAIR / 'datasets' / 'test.json', FAIR / 'datasets' / 'dev.json']
    hashes = {str(p): digest(read(p)) for p in protected}
    dest = RETRY / MODEL
    if not offline:
        (dest / 'calls').mkdir(parents=True, exist_ok=True)
        for record in failed.values():
            for trace in record['call_trace']:
                if trace['response_saved']:
                    original = source / trace['cache_file']
                    target = dest / trace['cache_file']
                    if target.exists():
                        assert digest(read(target)) == digest(read(original))
                    else:
                        shutil.copy2(original, target)
    frozen = FrozenInputs(FAIR)
    calibration = check_calibration(frozen, MODEL)
    calls = TrackedCalls(dest, MODEL, offline=offline, budget_cny=50)
    selected = [c for c in cases if c['key'] in failed]
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, c, 'test', frozen, calls, calibration): c['key'] for c in selected}
        for future in as_completed(futures):
            record = future.result()
            results[record['key']] = record
            print(record['key'], {m: s['status'] for m, s in record['method_status'].items()}, flush=True)
    assert hashes == {str(p): digest(read(p)) for p in protected}, 'historical_sources_changed'
    complete = all(all(s['status'] == 'ok' for s in r['method_status'].values()) for r in results.values())
    copied = {t['request_hash'] for r in failed.values() for t in r['call_trace'] if t['response_saved']}
    new_calls = [read(p) for p in (dest / 'calls').glob('*.json') if p.stem not in copied]
    old_events = [__import__('json').loads(line) for line in (source / 'failures.jsonl').read_text().splitlines()]
    tags = {t['tag'] for r in failed.values() for t in r['call_trace'] if 'request_failure' in t}
    manifest = dict(complete=complete, policy='Resume transport failures only; preserve every successful response and first valid grade; unchanged prompts/model/settings; no score-based retry.',
                    recovered_keys=sorted(results), original_record_hashes={k: digest(v) for k, v in failed.items()},
                    recovered_record_hashes={k: digest(v) for k, v in results.items()},
                    historical_source_hashes=hashes, new_saved_responses=len(new_calls),
                    new_prompt_tokens=sum(c['usage'].get('prompt_tokens', 0) for c in new_calls),
                    new_completion_tokens=sum(c['usage'].get('completion_tokens', 0) for c in new_calls),
                    original_failed_transport_attempts=sum(e['tag'] in tags for e in old_events),
                    missing_historical_failure_usage='unknown, not zero; original failures.jsonl retained',
                    new_failures_log=str(dest / 'failures.jsonl'))
    if offline:
        assert read(RETRY / 'manifest.json') == manifest
    else:
        dump(RETRY / 'manifest.json', manifest)
    assert complete, 'Some transport requests still failed; inspect retry logs before resuming.'
    load_completed_records(MODEL, cases, raw)
    print('Verified: all 350 inputs complete, successful judgments and historical files unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    run(args.workers, args.offline)

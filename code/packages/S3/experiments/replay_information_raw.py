"""从原始缓存复核 Information 核心分发、业务执行与能力探针；禁止网络。

只读研究数据；--manifest 将实际读取文件及导入源码的清单写到指定位置。
清单用于构建独立副本，不包含原始人工答卷或环境凭据。
"""
from pathlib import Path
import argparse
import hashlib
import json
import socket
import sys
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/results/review_revision_20260910'
READS = set()


def track(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    p = Path(args[0]).resolve()
    if not p.is_relative_to(ROOT):
        return
    mode = args[1] or ''
    if any(x in mode for x in ('w','a','+','x')):
        return
    if p.suffix in ('.json', '.py', '.md') and p.is_file():
        READS.add(p)


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def run():
    import analyze_contract_required_only as selection
    import retry_contract_transport as retry
    import review_fair_comparison_ids as parser
    from analyze_llm_reference_sensitivity import rescore
    from analyze_mechanism_reference_sensitivity import required_frame, measure
    import run_workflow_conditions_v2 as business
    import receiver_consequence_probe as probe

    raw,cases = selection.dataset('test')
    multi = [c for c in cases if c['split']=='multi']
    assert len(multi)==150
    refs = read(BASE/'unified_reference_20260922/references.json')
    checked_calls, checked_metrics = 0, 0
    for model in selection.MODELS:
        records = retry.load_completed_records(model, multi, raw)
        reference_rows = {r['key']:r for r in read(BASE/f'unified_reference_20260922/required_per_input/{model}.json')}
        mechanisms = read(BASE/f'unified_reference_20260922/mechanism_per_input/{model}.json')
        mechanism_rows = {m:{r['key']:r for r in rows} for m,rows in mechanisms.items()}
        for case, record in zip(multi,records):
            # Reparse literal responses, including locked-cell format repairs.
            outcomes = [record['method_outcomes']['joint'], *record['method_outcomes']['independent']]
            for outcome in outcomes:
                locked = {}
                for attempt in outcome['response_attempts']:
                    sha = attempt['request_hash']
                    candidates = [BASE/f'contract_transport_retry_20260912/{model}/calls/{sha}.json', BASE/f'fair_comparison_ids/{model}/calls/{sha}.json']
                    path = next(p for p in candidates if p.exists())
                    wire = read(path)
                    assert wire['content']==attempt['literal_response']
                    try:
                        parsed = parser.strict_json_response(wire['content'])
                        diagnostic = parser.ingest_entity_grades(parsed,outcome['expected_task_indexes'],record['memory_entity_ids'],locked)
                    except (ValueError,TypeError,KeyError):
                        diagnostic = None
                    checked_calls += 1
                assert outcome['status']=='ok' and diagnostic and not diagnostic['errors']
                grades = parser.validate_grades({'tasks':diagnostic['tasks']},outcome['expected_task_indexes'],len(record['memory_entity_ids']))
                assert grades==outcome['tasks']
            assert record['joint_tasks']==outcomes[0]['tasks']
            assert record['independent_tasks']==[t for o in outcomes[1:] for t in o['tasks']]
            for method in ('joint','independent'):
                metric = selection.evaluate(case,record,method)
                rescored = rescore(metric['visibility'],case['entities'],refs[case['key']])
                assert json.loads(json.dumps(rescored))==reference_rows[case['key']]['metrics'][method], (model,case['key'],method)
                actual = measure(case,required_frame(case,metric),refs[case['key']])
                assert json.loads(json.dumps(actual))==mechanism_rows[method][case['key']], (model,case['key'],method,'mechanism')
                checked_metrics += 1

    c = business.core
    fixture = read(c.OUT/'freeze.json')
    cases_b = {r['key']:r for r in read(c.OUT/'cases.json')}
    business_rows = 0
    for model in ('glm','gpt55_responses'):
        base = c.OUT/'formal'/model
        rows = [read(p) for p in sorted((base/'rows').glob('*.json'))]
        assert len(rows)==216
        for row in rows:
            c.audit_row(row,cases_b[row['case_key']],fixture,base)
        expected = read(base/'summary.json')
        assert c.aggregate(rows)==expected['overall']
        for level in c.LEVELS:
            for pattern in c.PATTERNS:
                for arm in ('D','U'):
                    subset = [r for r in rows if r['sharing']==level and r['pattern']==pattern and r['arm']==arm]
                    assert c.aggregate(subset)==expected['cells'][f'{level}/{pattern}'][arm]
        assert c.bootstrap(rows)==expected['paired_statistics']
        business_rows += len(rows)

    base = probe.OUT/'formal'
    protocol = read(base/'protocol.json')
    jobs = read(base/'jobs.json')
    source = {r['key']:r for r in read(probe.SOURCE/'formal/dataset.json')}
    rows = []
    for job in jobs:
        row = read(base/'rows'/f"{job['job_id']}.json")
        probe.verify_wire(base,job,row)
        wire = read(base/'receiver_glm/wire'/f"{row['request_hash']}.json")
        assert probe.score_response(source[row['key']],job,wire)==row['score']
        rows.append(row)
    assert len(rows)==384
    assert probe.compute_summary(rows,'formal',protocol)==read(base/'summary.json')
    return {'status':'PASS','core_inputs':150,'core_models':3,'core_method_input_checks':checked_metrics,
            'selection_response_attempts_reparsed':checked_calls,'business_workflow_arms_reexecuted':business_rows,
            'business_task_positions':business_rows*3,'probe_responses_rescored':len(rows),'network_calls':0,
            'scope':'Frozen parsed content and saved requests to scores and state checks; no semantic label validation or fresh model generation'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path)
    p.add_argument('--check-manifest',type=Path)
    a=p.parse_args()
    sys.dont_write_bytecode=True
    sys.addaudithook(track)
    def blocked(*args,**kwargs): raise RuntimeError('network disabled in raw replay')
    with ExitStack() as guard:
        guard.enter_context(patch.object(socket.socket,'connect',blocked))
        guard.enter_context(patch.object(socket,'create_connection',blocked))
        report=run()
    for module in list(sys.modules.values()):
        source=getattr(module,'__file__',None)
        if source:
            path=Path(source).resolve()
            if path.is_relative_to(ROOT) and path.suffix=='.py': READS.add(path)
    READS.add(Path(__file__).resolve())
    paths=sorted(READS)
    assert not any(p.name=='.env' or 'receipts' in p.parts for p in paths)
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest={'report':report,'files':files,'scope':'Local raw-cache replay snapshot; not publicly released; human answer sheets excluded'}
    if a.check_manifest:
        prior=read(a.check_manifest)
        assert prior['report']==report
        for name,sha in prior['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    if a.manifest:
        a.manifest.parent.mkdir(parents=True,exist_ok=True)
        a.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

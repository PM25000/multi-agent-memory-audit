"""显式事实 ID 的固定合同必要性诊断；与已中止数组输出协议独立。

每个模型每种方法只判断一次；阈值只控制后续确定性二值渲染。
不读参考答案来修复模型输出；离线模式没有客户端或联网回退。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from gpt_review_pilot import VERSION, build_request, digest, dump, json_response, read
from review_revision import EXP, OUT, RevisionCalls

FAIR_ROOT = OUT / 'fair_comparison_ids'
PROTOCOL_VERSION = 'fixed-contract-entity-grades-2'
SEED = 20260910
MAX_FORMAT_ATTEMPTS = 3
MAX_OUTPUT_TOKENS = 9000
MODELS = ('qwen', 'glm', 'gpt55_responses')
COMMON_PROMPT = '''你是任务信息必要性判断器。已提供固定任务合同、固定请求片段和按固定顺序排列的 Memory 事实。
任务合同 required 是本实验公开的输入要求，不是普遍业务或法律规则。不得新增或删除任务，不得修改请求片段、字段或事实。
判断每个 Memory 事实对每个要求输出的任务的必要性。所有任务的上下文均可见，但不能仅因另一任务需要某事实就把它判给目标任务。
只使用原文支持的主体、关系、用途和时间；不能从职业猜收入、从历史值猜当前值、从姓名推断属性。若固定请求片段已提供相同事实，应结合合同判断是否还需要该 Memory 事实，不得编造缺失信息。
等级统一定义：0=明确无关；1=仅有弱的可能相关性；2=对该任务有具体帮助，但并非合同明确必需；3=合同明确需要该事实所表达的值。
grades 是以 memory_entities 中真实 id 为键的 JSON 对象，每个真实 id 恰好对应一个整数 0、1、2 或 3。必须覆盖全部 Memory id，不能省略明确无关的事实，不能按最大 id 补不存在的编号。任务和对象键的输出顺序不限。不要输出事实值、解释、请求或新字段。
只输出 JSON：{"tasks":[{"task_index":0,"grades":{"e1":3,"e4":0,"e7":1,"e9":2}}]}。示例 id 仅示意格式，实际必须使用输入真实 id。
'''
JOINT_PROMPT = COMMON_PROMPT + '本次联合判断 contracts 中全部任务，tasks 必须覆盖每个 task_index 一次；程序按输入合同顺序规范化。'
INDEPENDENT_PROMPT = COMMON_PROMPT + '本次独立判断 target_index 指定的一个任务，tasks 仅包含该 task_index；其余合同仅作为相同上下文。'


def thresholds():
    return [i / 10 for i in range(41)]


def protocol_definition():
    """数据构建器在第一次模型调用前冻结的完整运行合同。"""
    return {
        'version': PROTOCOL_VERSION, 'seed': SEED,
        'joint_prompt': JOINT_PROMPT, 'independent_prompt': INDEPENDENT_PROMPT,
        'grades': [0, 1, 2, 3], 'thresholds': thresholds(),
        'retention_rule': 'grade + retention_u(key,task_index,entity_id) >= threshold',
        'retention_u': 'sha256 canonical-json {seed,key,task_index,entity_id}; first 13 hex digits / 2**52',
        'max_format_attempts': MAX_FORMAT_ATTEMPTS,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'wire_as_json': False,
        'wire_output': 'tasks:[{task_index:int,grades:{literal_memory_entity_id:integer_0_to_3}}]; task and key order may differ',
        'internal_output': 'Valid scores are normalized to arrays in fixed Memory and task order; no missing value is imputed.',
        'repair_policy': 'Initial plus at most two JSON/schema repairs. Lock each first valid task/entity grade. Repairs may fill missing or invalid cells; later changes to locked cells are overridden and audited. Unknown IDs/tasks and duplicate JSON keys are rejected. Duplicate task rows are rejected without locking new cells from that ambiguous task group. No score or reference is used.',
        'failure_policy': 'Execute joint and every independent target even if another path fails. complete means all paths terminal. A method with any failed path aborts the complete delivery transaction: no Memory or request sent to any recipient; operational P=0,L=0 with explicit validity rate. Both-valid conditional analysis is separate. Failed grades remain null, never imputed.',
        'input_fields': ['original', 'memory_entities', 'contracts'],
        'independent_additional_field': 'target_index',
        'test_requires_frozen_dev_calibration': True,
    }


def retention_u(key, task_index, eid):
    value = {'seed': SEED, 'key': key, 'task_index': task_index, 'entity_id': eid}
    return int(digest(value)[:13], 16) / (2 ** 52)


def retain_grade(grade, key, task_index, eid, threshold):
    if type(grade) is not int or not 0 <= grade <= 3:
        raise ValueError('invalid_grade')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 4:
        raise ValueError('invalid_threshold')
    return grade + retention_u(key, task_index, eid) >= threshold


def memory_entities(case):
    return [e for e in case['entities'] if e['source'] == 'memory']


def validate_case(case):
    if not isinstance(case.get('key'), str) or not case['key'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case['key']):
        raise ValueError('unsafe_or_empty_case_key')
    item = case['item']
    if not all(isinstance(item.get(k), str) for k in ('memory_context', 'user_input')):
        raise ValueError('invalid_original_input')
    entities = case['entities']
    ids = [e['id'] for e in entities]
    if any(not isinstance(eid, str) or not eid for eid in ids) or len(ids) != len(set(ids)):
        raise ValueError('invalid_entity_ids')
    for e in entities:
        if e['source'] not in ('memory', 'request'):
            raise ValueError('invalid_entity_source')
        source = item['memory_context' if e['source'] == 'memory' else 'user_input']
        if not e['value'] or not e['quote'] or e['quote'] not in source or e['value'] not in e['quote']:
            raise ValueError('nonliteral_entity_source')
    tasks = case['fixed_tasks']
    indexes = [t['task_index'] for t in tasks]
    if not tasks or any(type(i) is not int or i < 0 for i in indexes) or len(indexes) != len(set(indexes)):
        raise ValueError('invalid_fixed_task_indexes')
    for task in tasks:
        if not isinstance(task['name'], str) or not task['name']:
            raise ValueError('invalid_fixed_task_name')
        if not isinstance(task['required'], list) or any(not isinstance(f, str) or not f for f in task['required']) or len(task['required']) != len(set(task['required'])):
            raise ValueError('invalid_required_contract')
        qs = task['request_quotes']
        if not isinstance(qs, list) or any(not isinstance(q, str) or not q or q not in item['user_input'] for q in qs):
            raise ValueError('nonliteral_fixed_request')
    return case


def model_payload(case, target_index=None):
    """白名单组装，禁止把 reference 或 ground_truth 传给模型。"""
    validate_case(case)
    payload = {
        'original': {'memory': case['item']['memory_context'], 'request': case['item']['user_input']},
        'memory_entities': [{k: e[k] for k in ('id', 'label', 'value', 'source', 'quote')} for e in memory_entities(case)],
        'contracts': [{k: t[k] for k in ('task_index', 'name', 'request_quotes', 'required')} for t in case['fixed_tasks']],
    }
    if target_index is not None:
        if target_index not in [t['task_index'] for t in case['fixed_tasks']]:
            raise ValueError('unknown_target_task')
        payload['target_index'] = target_index
    return payload


def validate_grades(data, expected_indexes, entity_count):
    if not isinstance(data, dict) or set(data) != {'tasks'} or not isinstance(data['tasks'], list):
        raise ValueError('expected_only_tasks_array')
    rows = data['tasks']
    if any(not isinstance(r, dict) or set(r) != {'task_index', 'grades'} for r in rows):
        raise ValueError('expected_only_task_index_and_grades')
    if any(type(r['task_index']) is not int for r in rows) or [r['task_index'] for r in rows] != list(expected_indexes):
        raise ValueError(f'task_coverage_and_order_expected:{list(expected_indexes)}')
    for row in rows:
        grades = row['grades']
        if not isinstance(grades, list) or len(grades) != entity_count:
            raise ValueError(f'grade_length_expected:{entity_count}')
        if any(type(g) is not int or not 0 <= g <= 3 for g in grades):
            raise ValueError('grades_must_be_integers_0_to_3')
    return [{'task_index': r['task_index'], 'grades': list(r['grades'])} for r in rows]


def strict_json_response(raw):
    """重复 JSON 键不存在可审计的唯一赋值，不能采用解析器的最后值。"""
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_object)


def ingest_entity_grades(data, expected_indexes, entity_ids, locked):
    """累计首次合法格；错误仅触发格式修复，不调用任何效用/披露评分。"""
    expected_indexes = list(expected_indexes)
    entity_ids = list(entity_ids)
    allowed_tasks, allowed_ids = set(expected_indexes), set(entity_ids)
    errors, added, ignored = [], [], []
    if not isinstance(data, dict) or set(data) != {'tasks'}:
        errors.append('expected_only_tasks_array')
    rows = data.get('tasks') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        rows = []
        errors.append('tasks_must_be_array')
    counts = Counter(r.get('task_index') for r in rows if isinstance(r, dict) and type(r.get('task_index')) is int)
    duplicates = {i for i, count in counts.items() if count > 1}
    if duplicates:
        errors.append('duplicate_task_indexes:' + json.dumps(sorted(duplicates)))
    for row in rows:
        if not isinstance(row, dict):
            errors.append('task_row_must_be_object')
            continue
        if set(row) != {'task_index', 'grades'}:
            errors.append('expected_only_task_index_and_grades')
        index = row.get('task_index')
        if type(index) is not int or index not in allowed_tasks:
            errors.append('unknown_or_noninteger_task_index')
            continue
        if index in duplicates:
            # Neither repeated task row has precedence; lock no new value from
            # this ambiguous group, while retaining earlier unambiguous locks.
            continue
        grades = row.get('grades')
        if not isinstance(grades, dict):
            errors.append(f'grades_must_be_entity_id_object:task={index}')
            continue
        previous = locked.setdefault(index, {})
        for eid, grade in grades.items():
            if eid not in allowed_ids:
                errors.append(f'unknown_memory_id:task={index}:id={eid}')
                continue
            if type(grade) is not int or not 0 <= grade <= 3:
                errors.append(f'invalid_grade:task={index}:id={eid}')
                continue
            if eid in previous:
                if grade != previous[eid]:
                    ignored.append({'task_index': index, 'entity_id': eid,
                                    'locked_grade': previous[eid], 'attempted_grade': grade})
            else:
                previous[eid] = grade
                added.append({'task_index': index, 'entity_id': eid})
    missing = {str(index): [eid for eid in entity_ids if eid not in locked.get(index, {})]
               for index in expected_indexes}
    missing = {index: ids for index, ids in missing.items() if ids}
    if missing:
        errors.append('missing_entity_ids:' + json.dumps(missing, ensure_ascii=False, sort_keys=True))
    absent_tasks = [index for index in expected_indexes if index not in locked]
    if absent_tasks:
        errors.append('missing_task_indexes:' + json.dumps(absent_tasks))
    tasks = None if errors else [{'task_index': index, 'grades': [locked[index][eid] for eid in entity_ids]}
                                 for index in expected_indexes]
    return {'tasks': tasks, 'errors': list(dict.fromkeys(errors)),
            'added_cells': added, 'ignored_score_changes': ignored, 'missing_entity_ids': missing}


def materialize_tasks(case, graded_tasks, threshold):
    entities = memory_entities(case)
    indexes = [t['task_index'] for t in case['fixed_tasks']]
    rows = validate_grades({'tasks': graded_tasks}, indexes, len(entities))
    tasks = []
    for fixed, graded in zip(case['fixed_tasks'], rows):
        keep = [e['id'] for e, grade in zip(entities, graded['grades'])
                if retain_grade(grade, case['key'], fixed['task_index'], e['id'], threshold)]
        tasks.append({'name': fixed['name'], 'request_quotes': list(fixed['request_quotes']), 'keep_entity_ids': keep})
    return tasks, indexes


make_binary_tasks = materialize_tasks


class FrozenInputs:
    def __init__(self, root=FAIR_ROOT):
        self.root = Path(root)
        self.protocol_path = self.root / 'datasets' / 'protocol.json'
        self.protocol = read(self.protocol_path)
        if self.protocol.get('runner_contract') != protocol_definition():
            raise ValueError('frozen_runner_contract_mismatch')
        self.protocol_hash = digest(self.protocol)
        self.datasets = {split: read(self.root / 'datasets' / f'{split}.json') for split in ('dev', 'test')}
        self.dataset_hashes = {split: digest(data) for split, data in self.datasets.items()}
        if self.protocol.get('dataset_hashes') != self.dataset_hashes:
            raise ValueError('frozen_dataset_hash_mismatch')
        self.files = [self.protocol_path] + [self.root / 'datasets' / f'{split}.json' for split in ('dev', 'test')]
        for name, expected in self.protocol.get('scoring_files', {}).items():
            path=(EXP.parent/name).resolve()
            if not path.is_relative_to(EXP.parent.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('frozen_scoring_file_mismatch')
            self.files.append(path)
        self.raw_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.files}
        for data in self.datasets.values():
            if len({c['key'] for c in data}) != len(data):
                raise ValueError('duplicate_dataset_case_keys')
            for case in data:
                validate_case(case)

    def verify(self):
        for p in self.files:
            if hashlib.sha256(p.read_bytes()).hexdigest() != self.raw_hashes[str(p)]:
                raise ValueError('frozen_input_changed_during_run')


def check_calibration(frozen, model):
    path = frozen.root / 'calibration.json'
    if not path.exists():
        raise ValueError('test_requires_completed_dev_calibration')
    calibration = read(path)
    if calibration.get('protocol_hash') != frozen.protocol_hash or calibration.get('dev_dataset_hash') != frozen.dataset_hashes['dev']:
        raise ValueError('calibration_frozen_input_mismatch')
    if model not in calibration.get('models', {}):
        raise ValueError('calibration_model_missing')
    return digest(calibration)


def expected_model_spec(model):
    if model == 'gpt55_responses':
        active = read(EXP / 'active_models.json')['gpt']
        return {'model': active['model'], 'base_url': active['base_url'], 'key': active['credential_env'],
                'token_limit_parameter': 'max_completion_tokens', 'omit_temperature': True, 'api': 'responses'}
    from gpt_review_pilot import SUBJECTS
    return {'model': SUBJECTS[model]['model'], 'base_url': 'https://openrouter.ai/api/v1', 'key': 'OPENROUTER_API_KEY'}


class RecordedRequestFailure(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail['error'])


class TrackedCalls:
    """请求哈希逐次关联，线程局部 trace 不依赖并发缓存目录差集。"""
    def __init__(self, out, model, offline=False, budget_cny=250):
        self.out, self.offline = Path(out), offline
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / 'calls'
        self.cache.mkdir(exist_ok=True)
        self.subject = 'gpt' if model == 'gpt55_responses' else model
        self.spec = expected_model_spec(model)
        if model == 'gpt55_responses' and (self.spec['model'] != 'gpt-5.5' or self.spec.get('api') != 'responses' or 'openrouter' in self.spec['base_url']):
            raise ValueError('unauthorized_gpt_model_or_endpoint')
        config = {'subject': self.subject, 'model_key': model, 'spec': self.spec,
                  'wire_as_json': False, 'protocol_version': PROTOCOL_VERSION}
        config_path = self.out / 'model_config.json'
        if config_path.exists():
            if read(config_path) != config:
                raise ValueError('model_config_changed')
        elif offline:
            raise ValueError('offline_model_config_missing')
        else:
            dump(config_path, config)
        self.delegate = None
        if not offline:
            self.delegate = RevisionCalls(self.out, model=self.subject, budget_cny=budget_cny)
            if self.delegate.specs['gpt'] != self.spec:
                raise ValueError('model_environment_override_disallowed')
        self.local = threading.local()

    def start_case(self):
        self.local.trace = []

    @property
    def trace(self):
        return list(getattr(self.local, 'trace', []))

    def request(self, prompt, payload, tag, method, task_index, attempt):
        user = json.dumps(payload, ensure_ascii=False)
        args = build_request(self.spec, 'gpt', self.subject, prompt, user, MAX_OUTPUT_TOKENS, False)
        request_hash = digest({'version': VERSION, 'endpoint': self.spec['base_url'], 'args': args})
        path = self.cache / f'{request_hash}.json'
        failed_path = self.out / 'request_failures' / f'{request_hash}.json'
        trace_row = {'method': method, 'task_index': task_index, 'attempt': attempt,
            'tag': tag, 'request_hash': request_hash, 'cache_file': str(path.relative_to(self.out)),
            'schema_repair': attempt > 0, 'response_saved': False, 'usage': {},
            'elapsed_seconds': 0, 'response_id': None, 'response_model': None, 'finish_reason': None}
        self.local.trace.append(trace_row)
        if failed_path.exists():
            failure = read(failed_path)
            if failure['request_hash'] != request_hash:
                raise ValueError('request_failure_hash_mismatch')
            trace_row['request_failure'] = failure
            if path.exists():
                cached = read(path)
                trace_row.update({'response_saved': True, 'usage': cached.get('usage', {}),
                    'elapsed_seconds': cached.get('elapsed_seconds', 0), 'response_id': cached.get('response_id'),
                    'response_model': cached.get('response_model'), 'finish_reason': cached.get('finish_reason')})
            raise RecordedRequestFailure(failure)
        if not self.offline:
            dump(self.out / 'requests' / f'{request_hash}.json',
                 {'request_hash': request_hash, 'endpoint': self.spec['base_url'], 'request': args})
        if self.offline:
            if not path.exists():
                raise RuntimeError(f'offline_cache_miss:{tag}:{request_hash}')
            cached = read(path)
            if cached.get('request') != args or cached.get('endpoint') != self.spec['base_url']:
                raise ValueError('offline_cache_request_mismatch')
            if cached.get('finish_reason') != 'stop' or not cached.get('content', '').strip():
                raise ValueError('offline_cache_incomplete_response')
            raw = cached['content']
        else:
            try:
                raw = self.delegate.request('gpt', prompt, user, tag, max_tokens=MAX_OUTPUT_TOKENS, as_json=False)
                cached = read(path)
            except Exception as exc:
                failure = {'request_hash': request_hash, 'exception_type': type(exc).__name__, 'error': str(exc)[:300]}
                dump(failed_path, failure)
                trace_row['request_failure'] = failure
                if path.exists():
                    cached = read(path)
                    trace_row.update({'response_saved': True, 'usage': cached.get('usage', {}),
                        'elapsed_seconds': cached.get('elapsed_seconds', 0), 'response_id': cached.get('response_id'),
                        'response_model': cached.get('response_model'), 'finish_reason': cached.get('finish_reason')})
                raise RecordedRequestFailure(failure) from None
        trace_row.update({'response_saved': True, 'usage': cached.get('usage', {}),
            'elapsed_seconds': cached.get('elapsed_seconds', 0), 'response_id': cached.get('response_id'),
            'response_model': cached.get('response_model'), 'finish_reason': cached.get('finish_reason')})
        return raw


def call_grades(calls, case, method, target_index=None):
    payload = model_payload(case, target_index)
    indexes = [target_index] if target_index is not None else [t['task_index'] for t in case['fixed_tasks']]
    prompt = JOINT_PROMPT if method == 'joint' else INDEPENDENT_PROMPT
    request_payload = payload
    entity_ids = [e['id'] for e in memory_entities(case)]
    locked, responses = {}, []
    for attempt in range(MAX_FORMAT_ATTEMPTS):
        tag = f"{case['key']}:{method}" + (f'_{target_index}' if target_index is not None else '') + f':format_{attempt}'
        try:
            raw = calls.request(prompt, request_payload, tag, method, target_index, attempt)
        except Exception as exc:
            if calls.offline and not isinstance(exc, RecordedRequestFailure):
                raise
            failure = exc.detail if isinstance(exc, RecordedRequestFailure) else {'exception_type': type(exc).__name__, 'error': str(exc)[:300]}
            return {'status': 'transport_failure', 'method': method, 'task_index': target_index,
                    'expected_task_indexes': indexes, 'tasks': None, 'response_attempts': responses,
                    'error': failure}
        before = sum(len(grades) for grades in locked.values())
        try:
            data = strict_json_response(raw)
            diagnostic = ingest_entity_grades(data, indexes, entity_ids, locked)
        except (ValueError, TypeError, KeyError) as exc:
            diagnostic = {'tasks': None, 'errors': ['invalid_json' if isinstance(exc, json.JSONDecodeError) else str(exc)],
                          'added_cells': [], 'ignored_score_changes': [], 'missing_entity_ids': {}}
        metadata = {k: v for k, v in diagnostic.items() if k != 'tasks'}
        metadata.update({'attempt': attempt, 'request_hash': calls.trace[-1]['request_hash'],
            'literal_response': raw, 'locked_cell_count_before': before,
            'locked_cell_count_after': sum(len(grades) for grades in locked.values())})
        responses.append(metadata)
        if not diagnostic['errors']:
            tasks = validate_grades({'tasks': diagnostic['tasks']}, indexes, len(entity_ids))
            return {'status': 'ok', 'method': method, 'task_index': target_index,
                    'expected_task_indexes': indexes, 'tasks': tasks, 'response_attempts': responses}
        if attempt == MAX_FORMAT_ATTEMPTS - 1:
            return {'status': 'format_failure', 'method': method, 'task_index': target_index,
                    'expected_task_indexes': indexes, 'tasks': None, 'response_attempts': responses,
                    'error': 'format_attempts_exhausted:' + '|'.join(diagnostic['errors'])}
        request_payload = {'input': payload, 'invalid_response': raw,
            'validation_errors': diagnostic['errors'], 'repair_attempt': attempt + 1,
            'locked_grades': [{'task_index': index, 'grades': locked[index]} for index in indexes if index in locked],
            'instruction': '只修正 JSON/任务/id/整数格式。locked_grades 的合法判断已冻结，不能改变。可以只返回缺失或非法格的合法补充，也可返回完整对象；不得新增未知任务或未知 id。既有合法格的任何改动都会被程序恢复原值。输出 {"tasks":[{"task_index":整数,"grades":{"真实id":0到3整数}}]}；不要输出解释。'}
    raise AssertionError('unreachable')


def method_costs(trace):
    result = {}
    for method in ('joint', 'independent'):
        rows = [r for r in trace if r['method'] == method]
        unique = list({r['request_hash']: r for r in rows if r.get('response_saved', True)}.values())
        result[method] = {'logical_requests': len(rows), 'unique_responses': len(unique),
            'schema_repairs': sum(r['schema_repair'] for r in unique),
            'prompt_tokens': sum(r['usage'].get('prompt_tokens', r['usage'].get('input_tokens', 0)) or 0 for r in unique),
            'completion_tokens': sum(r['usage'].get('completion_tokens', r['usage'].get('output_tokens', 0)) or 0 for r in unique),
            'response_wait_seconds_sum': sum(r['elapsed_seconds'] for r in unique)}
    return result


def run_case(case, split, frozen, calls, calibration_hash=None):
    frozen.verify()
    if split == 'test' and digest(read(frozen.root / 'calibration.json')) != calibration_hash:
        raise ValueError('calibration_changed_during_test')
    calls.start_case()
    output = calls.out / 'cases' / f"{case['key']}.json"
    hashes = {'input_hash': digest(case), 'dataset_hash': frozen.dataset_hashes[split],
              'protocol_hash': frozen.protocol_hash, 'calibration_hash': calibration_hash}
    old = read(output) if output.exists() else None
    if old and (not old.get('complete') or any(old.get(k) != v for k, v in hashes.items())):
        raise ValueError('existing_case_frozen_inputs_mismatch')
    if old and not calls.offline:
        return old
    joint = call_grades(calls, case, 'joint')
    independent = [call_grades(calls, case, 'independent', task['task_index']) for task in case['fixed_tasks']]
    frozen.verify()
    if split == 'test' and digest(read(frozen.root / 'calibration.json')) != calibration_hash:
        raise ValueError('calibration_changed_during_test')
    joint_ok = joint['status'] == 'ok'
    independent_ok = all(outcome['status'] == 'ok' for outcome in independent)
    independent_failures = [outcome for outcome in independent if outcome['status'] != 'ok']
    independent_status = ('ok' if independent_ok else
        'transport_failure' if any(outcome['status'] == 'transport_failure' for outcome in independent_failures) else 'format_failure')
    method_status = {
        'joint': {'status': joint['status'], 'failed_task_indexes': [] if joint_ok else [t['task_index'] for t in case['fixed_tasks']],
                  'delivery_status': 'ready' if joint_ok else 'aborted'},
        'independent': {'status': independent_status, 'failed_task_indexes': [outcome['task_index'] for outcome in independent_failures],
                        'delivery_status': 'ready' if independent_ok else 'aborted'},
    }
    result = {'version': PROTOCOL_VERSION, 'wire_protocol': PROTOCOL_VERSION, 'complete': True,
        'completion_scope': 'all_predeclared_paths_terminal_not_all_outputs_valid',
        'key': case['key'], 'split': case['split'],
        'dataset_split': split, **hashes, 'memory_entity_ids': [e['id'] for e in memory_entities(case)],
        'fixed_tasks': case['fixed_tasks'], 'joint_tasks': joint['tasks'] if joint_ok else None,
        'independent_tasks': [task for outcome in independent for task in outcome['tasks']] if independent_ok else None,
        'method_status': method_status, 'method_outcomes': {'joint': joint, 'independent': independent},
        'call_trace': calls.trace, 'method_costs': method_costs(calls.trace)}
    if calls.offline:
        if old is None:
            raise ValueError('offline_expected_case_missing')
        if result != old:
            raise ValueError('offline_replay_case_mismatch')
    else:
        dump(output, result)
        if not joint_ok or not independent_ok:
            dump(calls.out / 'method_failures' / f"{case['key']}.json", {'key': case['key'], **hashes,
                 'method_status': method_status, 'method_outcomes': result['method_outcomes'], 'call_trace': calls.trace})
    return result


def run(model, split, workers=4, budget_cny=250, limit=None, offline=False, root=FAIR_ROOT):
    frozen = FrozenInputs(root)
    calibration_hash = check_calibration(frozen, model) if split == 'test' else None
    calls = TrackedCalls(Path(root) / model, model, offline, budget_cny)
    cases = frozen.datasets[split][:limit]
    progress_path = calls.out / f"{'offline_' if offline else ''}progress_{split}.json"
    failures = []
    completed = 0
    operationally_valid = 0
    method_failures = Counter()
    def progress():
        dump(progress_path, {'model': model, 'split': split, 'offline': offline, 'total': len(cases),
            'completed': completed, 'failed': len(failures), 'operationally_valid_cases': operationally_valid,
            'method_failures': dict(method_failures), 'protocol_hash': frozen.protocol_hash,
            'dataset_hash': frozen.dataset_hashes[split], 'calibration_hash': calibration_hash,
            'updated_utc': datetime.now(timezone.utc).isoformat()})
    progress()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, case, split, frozen, calls, calibration_hash): case['key'] for case in cases}
        for future in as_completed(futures):
            key = futures[future]
            try:
                record = future.result()
                completed += 1
                if all(status['status'] == 'ok' for status in record['method_status'].values()):
                    operationally_valid += 1
                for method, status in record['method_status'].items():
                    if status['status'] != 'ok':
                        method_failures[method] += 1
            except Exception as exc:
                failures.append({'key': key, 'exception_type': type(exc).__name__, 'error': str(exc)[:400]})
            progress()
            print(json.dumps({'model': model, 'split': split, 'completed': completed, 'failed': len(failures),
                  'method_failures': dict(method_failures), 'total': len(cases)}), flush=True)
    dump(calls.out / f"{'offline_' if offline else ''}failures_{split}.json", failures)
    if failures:
        raise RuntimeError(f'fair_comparison_failed_cases:{len(failures)}')
    return {'model': model, 'split': split, 'completed': completed, 'operationally_valid_cases': operationally_valid,
            'method_failures': dict(method_failures), 'offline': offline}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=(*MODELS, 'gpt'), required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--budget-cny', type=float, default=250)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1) or args.budget_cny <= 0:
        parser.error('workers, limit and budget must be positive')
    print(json.dumps(run('gpt55_responses' if args.model == 'gpt' else args.model, args.split,
          args.workers, args.budget_cny, args.limit, args.offline)), flush=True)

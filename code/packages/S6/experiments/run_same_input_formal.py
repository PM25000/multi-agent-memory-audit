"""Public read-only function projection; retired controller/intake omitted."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import run_same_input_independent as old
from same_input_native_api import SPECS, digest, read, encoded
from workflow_B_runtime import snapshot
from workflow_B_api import parse_wire_body, response_items, build_request
ROOT = old.ROOT

def dependencies():
    packages = ('pydantic', 'deepdiff', 'openai', 'httpx', 'python-dotenv', 'PyYAML', 'numpy', 'pandas', 'docstring-parser', 'cohere', 'google-genai')
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'package_versions': versions, 'agentdojo': 'vendored-source-179'}

def local_import_closure(paths):
    """Freeze syntactic local imports, including imports inside function bodies.

    External distributions are pinned separately by runtime version metadata.
    Dynamic native loading is covered by the original 179-file manifest.
    """
    pending = [Path(p).resolve() for p in paths if Path(p).suffix == '.py']
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                candidate = ROOT / 'experiments' / (name.split('.')[0] + '.py')
                if candidate.exists() and candidate.resolve() not in found:
                    pending.append(candidate.resolve())
    return found

def verify_wire_fields(response, model_key):
    """Reproduce transport parsing, retaining genuinely malformed receipts.

    Only parsing is caught. Comparisons are outside the catch, so a purported
    invalid_response cannot hide drift in successfully extracted wire fields.
    """
    if response.get('raw_body') is None:
        allowed = ('input_cap_exceeded', 'transport_failure', 'client_preflight_failure')
        if response['status'] not in allowed:
            raise ValueError('No wire body for purported parsed response')
        expected_attempted = response['status'] == 'transport_failure'
        if response.get('api_attempted') is not expected_attempted:
            raise ValueError('Body-free failure has inconsistent API-attempt accounting')
        if response.get('raw_response') is not None or response.get('parsed') is not None or response.get('usage'):
            raise ValueError('Body-free failure contains fabricated model output')
        if 'returned_model' in response or 'framing' in response:
            raise ValueError('Body-free failure contains fabricated identity/framing')
        return
    if response.get('api_attempted') is not True:
        raise ValueError('Provider raw body incorrectly counted as no API attempt')
    fields = {'raw_response': None, 'usage': {}, 'parsed': None}
    parse_error = None
    try:
        raw, framing = parse_wire_body(response['raw_body'], responses=False)
        if not isinstance(raw, dict):
            raise ValueError('response_envelope_not_object')
        fields.update(raw_response=raw, framing=framing, usage=raw.get('usage') or {}, returned_model=raw.get('model'))
        if not raw.get('model'):
            expected = 'model_identity_missing'
        elif raw['model'] not in SPECS[model_key]['aliases']:
            expected = 'model_identity_mismatch'
        else:
            choices = raw.get('choices')
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('exactly_one_native_choice_required')
            message = choices[0].get('message') if isinstance(choices[0], dict) else None
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                raise ValueError('native_response_role_must_be_assistant')
            parsed = response_items(raw, responses=False)
            fields['parsed'] = parsed
            expected = 'incomplete_response' if not parsed['complete'] else 'invalid_native_tool_response' if parsed['format_errors'] else 'ok'
    except Exception as exc:
        expected, parse_error = ('invalid_response', type(exc).__name__)
    for key, value in fields.items():
        if response.get(key) != value:
            raise ValueError('Wire field differs: ' + key)
    for key in ('framing', 'returned_model'):
        if key not in fields and key in response:
            raise ValueError('Field fabricated before parse success: ' + key)
    if response['status'] != expected:
        raise ValueError('Raw-derived status differs')
    if parse_error is not None and response.get('error_type') != parse_error:
        raise ValueError('Original parser error type differs')

def role_seed(w, role, ids, loaded, prior, client, repeat, arm):
    """Reconstruct the exact old helper DTO to verify an unfinished journal.

    This is only a checker; it is never a second execution prompt builder.
    """
    dynamic = [{'receiver': dep, 'tool_results': [{'function': t['function'], 'result': t['result'], 'error': t['error']} for t in prior[dep]['tool_traces']]} for dep in role.get('dependencies', [])]
    user = {'original_workflow_request': w['original_prompt'], 'assigned_task': role['instruction'], 'source_visibility_contract': old.RECEIVER, 'reference_material': [old.visible(x) for x in w['records'] if x['id'] in ids]}
    if dynamic:
        user['actual_predecessor_tool_results'] = dynamic
    history = [{'role': 'system', 'content': old.SYSTEM}, {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)}]
    schema = loaded.tools_schema(role['tools'])
    row = {'id': role['id'], 'visible_ids': ids, 'input': deepcopy(history), 'schemas': schema, 'dynamic_results': dynamic, 'rounds': [], 'tool_traces': [], 'text': [], 'termination': None, 'response_complete': True, 'input_delivery_evidence': 'not_attempted', 'rejected_calls': []}
    initial_state = snapshot(loaded.env)
    binding = digest({'workflow': w, 'model': client.config, 'repeat': repeat, 'arm': arm, 'role': role, 'input': history, 'schemas': schema, 'initial_state': initial_state})
    return (history, schema, row, initial_state, binding)

def bind_request(client, logical, phase, history, schema, linked):
    key = hashlib.sha256(logical.encode('utf8')).hexdigest() + '.json'
    request = build_request(client.spec, history, schema, max_tokens=old.TOKENS)
    expected = {'logical_id': logical, 'phase': phase, 'model_key': client.model, 'request': request, 'request_sha256': digest(request), 'request_bytes': len(encoded(request)), 'scope_configuration_sha256': digest(client.config)}
    for group in ('intents', 'preflight_rejections'):
        path = client.root / group / key
        if path.exists():
            stored = read(path)
            if any((stored.get(k) != v for k, v in expected.items())):
                raise ValueError('Persisted request is not the actual public history: ' + group)
            linked.add((group, key))
            receipt = client.root / 'responses' / key
            if receipt.exists():
                response = read(receipt)
                if any((response.get(k) != expected[k] for k in ('logical_id', 'phase', 'model_key', 'request_sha256'))):
                    raise ValueError('Receipt logical phase/model differs')
                linked.add(('responses', key))

def role_request_links(w, role, ids, loaded, prior, client, repeat, arm, folder, linked):
    history, schema, row, initial_state, binding = role_seed(w, role, ids, loaded, prior, client, repeat, arm)
    journal = folder / 'journal' / arm / role['id']
    visited = set()
    for turn in range(old.TURNS):
        logical = f"{w['id']}/repeat{repeat}/{arm}/{role['id']}/turn{turn}"
        bind_request(client, logical, 'executor', history, schema, linked)
        path = journal / f'turn{turn}.json'
        if not path.exists():
            break
        visited.add(path.name)
        cp = read(path)
        history = cp['history']
        if cp['row']['termination']:
            break
    actual = {p.name for p in journal.glob('*.json')} - {'complete.json'}
    if actual != visited:
        raise ValueError('Role journal contains a gap or post-termination turn')

def selector_seed(w):
    ids = [r['id'] for r in w['records']]
    rids = [r['id'] for r in w['receivers']]
    info = {'workflow_request': w['original_prompt'], 'receivers': [{'id': r['id'], 'task': r['instruction']} for r in w['receivers']], 'records': [old.visible(r) for r in w['records']], 'source_visibility_contract': old.SELECTOR}
    schema = [{'type': 'function', 'function': {'name': 'select_records', 'description': 'Select record IDs separately for every receiver.', 'parameters': {'type': 'object', 'properties': {rid: {'type': 'array', 'items': {'type': 'string', 'enum': ids}, 'uniqueItems': True} for rid in rids}, 'required': rids, 'additionalProperties': False}}}]
    history = [{'role': 'system', 'content': old.SELECT_SYSTEM}, {'role': 'user', 'content': json.dumps(info, ensure_ascii=False)}]
    return (info, schema, history)

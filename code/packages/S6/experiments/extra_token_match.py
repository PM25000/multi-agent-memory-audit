"""Offline, bounded full Chat prompt matching; never calls a model API.

Official pinned tokenizer assets are distinct from provider token accounting.
Live pairs must retain both receipts and pass provider_equal() separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'experiments/tokenizers/execution_extension_20261005'
OFFICIAL = {
    'qwen': ('Qwen/Qwen3.6-27B', '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'),
    'glm': ('zai-org/GLM-5.2', 'cf457fa734ab149ffef225f80893eb38c6ff5cdc'),
}
FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def request_sha(request):
    from same_input_native_api import digest
    return digest(request)


def fetch_assets(model):
    """Explicit public asset acquisition, no key, no remote Python code."""
    import requests
    repo, revision = OFFICIAL[model]
    dest = ASSETS / model
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in FILES:
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        payload = response.content
        if name.endswith('.json'):
            json.loads(payload)
        path = dest / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError('Pinned tokenizer asset drift')
        if not path.exists():
            path.write_bytes(payload)
        rows.append({'file': name, 'sha256': sha(payload), 'bytes': len(payload), 'source_url': url})
    manifest = {'repo': repo, 'revision': revision, 'files': rows,
                'trust_remote_code': False, 'thinking': False,
                'count_scope': 'official rendered messages, tools, assistant generation prompt',
                'tool_argument_adapter': 'Chat wire JSON strings parsed to dict for official template only',
                'provider_token_equality_requires_receipts': True}
    target = dest / 'manifest.json'
    body = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    if target.exists() and json.loads(target.read_text(encoding='utf8')) != manifest:
        raise ValueError('Tokenizer manifest drift')
    if not target.exists():
        target.write_text(body, encoding='utf8')
    return manifest


class FullPromptCounter:
    def __init__(self, model):
        from transformers import AutoTokenizer
        self.model = model
        self.path = ASSETS / model
        self.manifest = json.loads((self.path / 'manifest.json').read_text(encoding='utf8'))
        if (self.manifest['repo'], self.manifest['revision']) != OFFICIAL[model]:
            raise ValueError('Unapproved tokenizer revision')
        for row in self.manifest['files']:
            if sha((self.path / row['file']).read_bytes()) != row['sha256']:
                raise ValueError('Tokenizer asset hash mismatch')
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True,
                                                       trust_remote_code=False)

    def count_request(self, request):
        """Count full conversational input, never serialized outer JSON bytes.

        Parameters such as output budget and temperature are not prompt text.
        The provider may nevertheless use a different template/adapter.
        """
        if 'messages' not in request or 'input' in request:
            raise ValueError('Only native Chat requests are supported')
        history = deepcopy(request['messages'])
        for message in history:
            for call in message.get('tool_calls') or []:
                function = call.get('function', call)
                if isinstance(function.get('arguments'), str):
                    function['arguments'] = json.loads(function['arguments'])
                if not isinstance(function.get('arguments'), dict):
                    raise ValueError('Tool arguments must parse to dictionary')
        tokens = self.tokenizer.apply_chat_template(history, tools=deepcopy(request.get('tools', [])),
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=False)
        return len(tokens)

    def count(self, history, schemas, *, max_tokens=2400):
        from same_input_native_api import SPECS
        from workflow_B_api import build_request
        request = build_request({'model': SPECS[self.model]['model']}, history, schemas, max_tokens)
        return self.count_request(request)


def match_padding(counter, target_history, candidate_history, schemas, *, target_tokens=None,
                  max_tokens=2400, max_evaluations=384, max_units=32768,
                  marker_prefix='__token_padding'):
    """Replace only the text of one explicitly marked reference-material slot.

    The candidate must already contain the desired genuine records and exactly
    one synthetic slot. This neither selects sources nor consults outcomes.
    Bounded binary proposal plus finite neighbors; equality is asserted on the
    complete rebuilt prompt. Search failure is retained, never an API retry.
    """
    from same_input_native_api import SPECS
    from workflow_B_api import build_request
    if not 1 <= max_evaluations <= 512 or not 1 <= max_units <= 65536:
        raise ValueError('Search caps must be explicit and bounded')
    spec = {'model': SPECS[counter.model]['model']}
    target_request = None if target_history is None else build_request(
        spec, deepcopy(target_history), deepcopy(schemas), max_tokens)
    if target_tokens is None:
        if target_request is None:
            raise ValueError('A target history or explicit token budget is required')
        target = counter.count_request(target_request)
    else:
        if type(target_tokens) is not int or target_tokens <= 0:
            raise ValueError('Token budget must be a positive integer')
        target = target_tokens
    candidate = deepcopy(candidate_history)
    slots = []
    for index, message in enumerate(candidate):
        if message.get('role') != 'user' or not isinstance(message.get('content'), str):
            continue
        try:
            content = json.loads(message['content'])
        except ValueError:
            continue
        for number, record in enumerate(content.get('reference_material', [])):
            if isinstance(record.get('id'), str) and record['id'].startswith(marker_prefix):
                if record.get('source') != {} or not str(record.get('text', '')).startswith('NON_EVIDENCE '):
                    raise ValueError('Padding marker must not carry source metadata')
                slots.append((index, content, number))
    if len(slots) != 1:
        raise ValueError('Exactly one explicit token padding slot required')
    index, payload, slot = slots[0]
    observations = []
    seen = {}

    def evaluate(units):
        if units in seen:
            return seen[units]
        if len(observations) >= max_evaluations:
            return None
        payload['reference_material'][slot]['text'] = 'NON_EVIDENCE ' + ' x' * units
        candidate[index]['content'] = json.dumps(payload, ensure_ascii=False)
        request = build_request(spec, candidate, schemas, max_tokens)
        value = counter.count_request(request)
        observations.append({'units': units, 'tokens': value})
        seen[units] = value
        return value

    found = None
    base = evaluate(0)
    if base == target:
        found = 0
    elif base < target:
        lo, hi = 0, min(max_units, max(1, target - base))
        while len(observations) < max_evaluations:
            value = evaluate(hi)
            if value == target:
                found = hi
                break
            if value >= target or hi == max_units:
                break
            lo, hi = hi, min(max_units, hi * 2)
        if found is None:
            while hi - lo > 1 and len(observations) < max_evaluations:
                middle = (lo + hi) // 2
                value = evaluate(middle)
                if value == target:
                    found = middle
                    break
                if value < target:
                    lo = middle
                else:
                    hi = middle
            if found is None:
                # Proposals need not be monotone; only explicit equality passes.
                for units in range(max(0, lo - 128), min(max_units, hi + 128) + 1):
                    value = evaluate(units)
                    if value == target:
                        found = units
                        break
                    if len(observations) >= max_evaluations:
                        break
    record = {'status': 'local_match' if found is not None else 'local_match_unavailable',
              'model': counter.model, 'target_tokens': target, 'candidate_base_tokens': base,
              'max_evaluations': max_evaluations, 'max_units': max_units,
              'evaluations': observations, 'units': found,
              'target_request_sha256': request_sha(target_request) if target_request is not None else None,
              'tokenizer_revision': counter.manifest['revision'],
              'provider_equality': 'unverified', 'real_model_calls': 0}
    if found is None:
        return None, record
    # Rebuild chosen payload, since the last searched proposal may differ.
    payload['reference_material'][slot]['text'] = 'NON_EVIDENCE ' + ' x' * found
    candidate[index]['content'] = json.dumps(payload, ensure_ascii=False)
    request = build_request(spec, candidate, schemas, max_tokens)
    assert counter.count_request(request) == target
    record.update(candidate_tokens=target, candidate_request_sha256=request_sha(request))
    return candidate, record


def provider_equal(target_receipt, candidate_receipt):
    """No outcome-based retry: missing/invalid usage is an explicit mismatch."""
    values = [row.get('usage', {}).get('prompt_tokens') for row in (target_receipt, candidate_receipt)]
    valid = all(type(value) is int and value >= 0 for value in values)
    return {'status': 'provider_match' if valid and values[0] == values[1] else
                    'provider_mismatch' if valid else 'provider_count_missing',
            'target_prompt_tokens': values[0], 'candidate_prompt_tokens': values[1],
            'provider_equal': valid and values[0] == values[1]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['fetch', 'selfcheck'])
    parser.add_argument('--model', choices=OFFICIAL, required=True)
    args = parser.parse_args()
    if args.operation == 'fetch':
        print(json.dumps(fetch_assets(args.model), ensure_ascii=False))
    else:
        counter = FullPromptCounter(args.model)
        target = [{'role': 'system', 'content': 'Ignore __token_padding NON_EVIDENCE records.'},
                  {'role': 'user', 'content': json.dumps({'assigned_task': 'Return source only',
                      'reference_material': [{'id': 'known', 'text': 'the correct fixture answer'}]})}]
        candidate = deepcopy(target)
        payload = json.loads(candidate[1]['content'])
        payload['reference_material'] = [{'id': '__token_padding_0', 'source': {}, 'text': 'NON_EVIDENCE '}]
        candidate[1]['content'] = json.dumps(payload)
        # Give target a substantial extra source payload so padding can grow.
        original = json.loads(target[1]['content'])
        original['reference_material'][0]['text'] += ' public fixture' * 50
        target[1]['content'] = json.dumps(original)
        schemas = [{'type': 'function', 'function': {'name': 'echo', 'description': 'Return argument',
                    'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}}]
        matched, audit = match_padding(counter, target, candidate, schemas)
        assert matched is not None and audit['status'] == 'local_match'
        # Real Chat role histories contain JSON-string tool-call arguments.
        candidate.extend([{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'mock_call', 'type': 'function', 'function': {'name': 'echo',
                'arguments': '{"text":"fixture"}'}}]},
            {'role': 'tool', 'tool_call_id': 'mock_call', 'content': '{"result":"fixture"}'}])
        second, full_history_audit = match_padding(counter, None, candidate, schemas, target_tokens=8192)
        assert second is not None and counter.count(second, schemas) == 8192
        assert second[0] == candidate[0] and second[2:] == candidate[2:]
        assert json.loads(second[1]['content'])['assigned_task'] == 'Return source only'
        impossible, rejected = match_padding(counter, None, candidate, schemas, target_tokens=1)
        assert impossible is None and rejected['status'] == 'local_match_unavailable'
        assert provider_equal({'usage': {'prompt_tokens': 9}}, {'usage': {'prompt_tokens': 9}})['provider_equal']
        assert not provider_equal({'usage': {'prompt_tokens': 9}}, {'usage': {'prompt_tokens': 10}})['provider_equal']
        assert not provider_equal({'usage': {}}, {'usage': {'prompt_tokens': 9}})['provider_equal']
        print(json.dumps({'first_history': audit, 'tool_history': full_history_audit,
                          'under_budget_rejection': rejected, 'provider_gate_controls': 'pass'},
                         ensure_ascii=False))

"""Conservative request-fact visibility, with source spans and unresolved cases.

This module does not edit annotations, infer semantic equivalence, or call a model.
Legacy quote visibility remains replayable. Repairs only remove discourse framing
or retain a complete, explicitly current first-person sentence from a long quote.
Anything needing semantic adjudication remains unknown, not a new correct label.
"""
from __future__ import annotations

import re
from review_value_matching import NUMERIC, normalize_value, source_value_match

VISIBILITY_VERSION = 'request-source-span-conservative-2'
LEGACY_VERSION = 'request-full-quote-1'
PUNCTUATION = ' \t\r\n，,。.;；'
# Connectives need punctuation or an explicit following grammatical boundary.
# In particular, 先前 and 最后一班 must never become 前 and 一班.
DISCOURSE = re.compile(
    r'(?:(?:另外|此外|对了|然后|首先|其次|最后)[，,：:]|'
    r'(?:另外|此外|然后|顺便|首先|其次|最后|先)'
    r'(?=(?:我(?:想|打算|计划|准备|要|寻思)|你|请|帮我|替我|给我|订|安排|预约|查询|查一下|办理|把|听说|最近|这个月|孩子说)))')
SENTENCE = re.compile(r'[^。！？；\n]+[。！？；\n]?')
SELF_CURRENT = re.compile(r'^我(?:打算|计划|想|准备|要|将|会|需要|决定|希望)')
HISTORY = re.compile(r'历史|过去|去年|上次|曾经|既往|之前')
# Narrow, explicitly enumerated framing grammar; arbitrary prior sentences are
# not dropped. Dates here are scheduling framing, so this rule is restricted to
# a current user's destination fact, never a date, other actor or condition.
CURRENT_DESTINATION = re.compile(r'^(?:用户)?(?:当前|本次)(?:行程|旅行|旅游|出行|租车)?目的地(?:城市|国家)?$')
SCHEDULING_FRAME = re.compile(
    r'^(?:我)?想趁(?:这周|本周|下周|周末|本周末|下周末)(?:的)?'
    r'(?:调休|休假|放假|休息)(?:把)?(?:这)?(?:几|两|三|四|五|六|七|八|九|十|[2-9])件事'
    r'(?:一块儿|一起)(?:办了|办完|处理完)[。；]?$')


def occurrences(text, needle):
    """Overlapping exact positions; no normalization can change source offsets."""
    if not needle:
        return []
    return [m.start() for m in re.finditer(r'(?=' + re.escape(needle) + r')', text)]


def boundary_valid_at(value, text, start):
    """Check the selected occurrence, not an unrelated equal value elsewhere."""
    if text[start:start+len(value)] != value:
        return False
    normalized = normalize_value(text)
    v = normalize_value(value)
    pos = len(normalize_value(text[:start]))
    end = pos + len(v)
    if v[0].isascii() and v[0].isalnum() and pos and normalized[pos-1].isascii() and normalized[pos-1].isalnum():
        return False
    if v[-1].isascii() and v[-1].isalnum() and end < len(normalized) and normalized[end].isascii() and normalized[end].isalnum():
        return False
    numbers = {(m.start(), m.end(), m.group()) for m in NUMERIC.finditer(normalized)}
    return all((pos+m.start(), pos+m.end(), m.group()) in numbers for m in NUMERIC.finditer(v))


def legacy_available_facts(task, facts, broadcast_ids=None):
    kept = set(task['keep_entity_ids'] if broadcast_ids is None else broadcast_ids)
    request = '\n'.join(task['request_quotes'])
    kept.update(e['id'] for e in facts if e['source'] == 'request' and e['quote'] in request)
    return kept


def trim_discourse(text):
    left = len(text) - len(text.lstrip(PUNCTUATION))
    end = len(text.rstrip(PUNCTUATION))
    while left < end:
        prefix = DISCOURSE.match(text, left)
        if prefix is None:
            break
        left = prefix.end()
        while left < end and text[left] in PUNCTUATION:
            left += 1
    return left, end


def support_candidates(fact):
    """Generate candidates without looking at a method's output or score."""
    quote, value = fact['quote'], fact['value']
    left, end = trim_discourse(quote)
    if quote[left:end] != quote and value in quote[left:end]:
        yield left, end, 'discourse_trim'
    segments = list(SENTENCE.finditer(quote))
    if len(segments) < 2 or HISTORY.search(fact['label']):
        return
    if not CURRENT_DESTINATION.fullmatch(fact['label']):
        return
    value_segments = [s for s in segments if value in s.group()]
    if len(value_segments) != 1:
        return
    segment = value_segments[0]
    if segment != segments[-1] or any(not SCHEDULING_FRAME.fullmatch(s.group().strip()) for s in segments[:-1]):
        return
    a, b = trim_discourse(segment.group())
    support = segment.group()[a:b]
    if SELF_CURRENT.match(support):
        yield segment.start()+a, segment.start()+b, 'self_contained_current_sentence'


def request_fact_visibility(fact, request_quotes, source_request):
    """Return present/absent/unknown plus checkable source and output evidence.

    `unknown` is a lexical candidate only, never a semantically validated positive.
    Source positions are zero-based half-open Python character offsets.
    """
    if fact['source'] != 'request':
        raise ValueError('request_fact_required')
    quote, value = fact['quote'], fact['value']
    if not quote or not value or quote not in source_request or value not in quote:
        raise ValueError('ungrounded_request_fact')
    if any(q not in source_request for q in request_quotes):
        raise ValueError('output_request_not_verbatim_source')
    if quote in '\n'.join(request_quotes):
        return {'status': 'present', 'reason': 'legacy_full_quote', 'evidence': quote}
    literal_indexes = [i for i, q in enumerate(request_quotes) if source_value_match(value, q)]
    if not literal_indexes:
        return {'status': 'absent', 'reason': 'no_boundary_valid_literal_value'}
    q_positions = occurrences(source_request, quote)
    v_positions = occurrences(quote, value)
    if len(q_positions) != 1 or len(v_positions) != 1:
        return {'status': 'unknown', 'reason': 'ambiguous_fact_source_position',
                'candidate_output_indexes': literal_indexes}
    fact_value_start = q_positions[0] + v_positions[0]
    if not boundary_valid_at(value, source_request, fact_value_start):
        return {'status': 'unknown', 'reason': 'fact_value_source_boundary_unresolved',
                'candidate_output_indexes': literal_indexes}
    for left, end, rule in support_candidates(fact):
        support = quote[left:end]
        # A bare value cannot establish its actor, time, purpose or relation.
        context = support.replace(value, '', 1).strip(PUNCTUATION)
        if len(context) < 2 or not source_value_match(value, support):
            continue
        expected_start = q_positions[0] + left
        for index in literal_indexes:
            output = request_quotes[index]
            o_positions = occurrences(source_request, output)
            # An output span with more than one possible provenance is unresolved.
            if len(o_positions) != 1:
                continue
            local = expected_start - o_positions[0]
            if local < 0 or output[local:local+len(support)] != support:
                continue
            if not boundary_valid_at(value, output, fact_value_start-o_positions[0]):
                continue
            return {'status': 'present', 'reason': rule, 'evidence': support,
                    'source_span': [expected_start, expected_start+len(support)],
                    'source_value_span': [fact_value_start, fact_value_start+len(value)],
                    'output_index': index, 'output_span': [local, local+len(support)]}
    return {'status': 'unknown', 'reason': 'value_seen_semantic_context_unresolved',
            'candidate_output_indexes': literal_indexes,
            'source_value_span': [fact_value_start, fact_value_start+len(value)]}


def available_facts(task, facts, broadcast_ids=None, *, source_request, possible=False):
    kept = set(task['keep_entity_ids'] if broadcast_ids is None else broadcast_ids)
    for e in facts:
        if e['source'] != 'request':
            continue
        result = request_fact_visibility(e, task['request_quotes'], source_request)
        if result['status'] == 'present' or possible and result['status'] == 'unknown':
            kept.add(e['id'])
    return kept


def score_visible(tasks, mapping, facts, ref, visibility_fn, broadcast=False):
    """Same fixed denominators and recipient counting as the historical scorer."""
    union = set().union(*(set(t['keep_entity_ids']) for t in tasks))
    bytask, actual = {}, []
    for i, t in enumerate(tasks):
        gi = mapping[i]
        visible = visibility_fn(t, facts, union if broadcast else None)
        bytask.setdefault(gi, set()).update(visible)
        actual.append({'block_index': i, 'task_index': gi, 'visible': sorted(visible)})
    fields = []
    for r in ref['bindings']:
        eligible = r['status'] == 'available' and ref['tasks'][r['task_index']]['active']
        correct = eligible and set(r['entity_ids']) <= bytask.get(r['task_index'], set())
        fields.append({'task_index': r['task_index'], 'field': r['field'],
                       'eligible': eligible, 'correct': bool(correct), 'reference_status': r['status']})
    memory = {e['id'] for e in facts if e['source'] == 'memory'}
    lower, upper, all_lower, all_upper, transmission = [], [], [], [], 0
    for r in actual:
        i, v = r['task_index'], set(r['visible'])
        definite = set(ref['needed'].get(str(i), []))
        possible = set(ref['possibly_needed'].get(str(i), []))
        if i >= 0 and not ref['tasks'][i]['active']:
            definite, possible = set(), set()
        transmission += len(v & memory)
        lower.extend((r['block_index'], e) for e in (v & memory) - possible)
        upper.extend((r['block_index'], e) for e in (v & memory) - definite)
        all_lower.extend((r['block_index'], e) for e in v - possible)
        all_upper.extend((r['block_index'], e) for e in v - definite)
    eligible_n, correct_n = sum(r['eligible'] for r in fields), sum(r['correct'] for r in fields)
    return {'correct': correct_n, 'total': len(fields), 'eligible_total': eligible_n,
            'P_all': correct_n/len(fields) if fields else None,
            'P_available': correct_n/eligible_n if eligible_n else None,
            'L_lower': len(lower), 'L_upper': len(upper),
            'L_all_lower': len(all_lower), 'L_all_upper': len(all_upper),
            'memory_transmissions': transmission, 'unique_L_lower': len({e for _, e in lower}),
            'fields': fields, 'visibility': actual,
            'leak_pairs_lower': sorted(lower), 'leak_pairs_upper': sorted(upper)}


def controlled_score(tasks, mapping, facts, ref, broadcast=False, *, source_request, possible=False):
    return score_visible(tasks, mapping, facts, ref,
                         lambda task, fs, ids: available_facts(task, fs, ids,
                             source_request=source_request, possible=possible), broadcast)

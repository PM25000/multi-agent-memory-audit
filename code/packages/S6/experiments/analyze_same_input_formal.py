"""Public read-only function projection; retired controller/intake omitted."""
from __future__ import annotations
from collections import Counter

def usage_accounting(receipts, orphan_intents, preflight_rejections):
    """Costs from distinct raw receipts, not planned nodes or per-arm selectors."""
    attempted = [r for r in receipts if r.get('api_attempted') is True]
    fields = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens')
    tokens, missing = ({}, {})
    for field in fields:
        values = []
        for row in attempted:
            u = row.get('usage') or {}
            v = (u.get('completion_tokens_details') or {}).get('reasoning_tokens', u.get('reasoning_tokens')) if field == 'reasoning_tokens' else u.get(field)
            if type(v) is int and v >= 0:
                values.append(v)
        tokens[field] = sum(values)
        missing[field] = len(attempted) - len(values)
    return {'recorded_receipts': len(receipts), 'receipt_status_counts': dict(Counter((r['status'] for r in receipts))), 'ok_receipts': sum((r['status'] == 'ok' for r in receipts)), 'non_ok_receipts': sum((r['status'] != 'ok' for r in receipts)), 'confirmed_attempts_with_receipt': len(attempted), 'intent_without_receipt_count': len(orphan_intents), 'attempts_interval_from_logs': [len(attempted), len(attempted) + len(orphan_intents)], 'input_cap_preflight_rejections': len(preflight_rejections), 'recorded_tokens': tokens, 'attempted_receipts_missing_token_field': missing, 'attempted_receipts_with_no_usage': sum((not r.get('usage') for r in attempted)), 'price_or_currency_inferred': False, 'wall_times_added_as_latency': False}

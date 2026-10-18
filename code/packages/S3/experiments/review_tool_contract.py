"""Pure parsing and structural checks copied from the historical tool experiment.

Kept separate so offline evaluation does not import API clients or load .env.
"""
import json
import re

PLACEHOLDER_RE = re.compile(r'\[[^\]]*\]|\*{2,}|<[^>]*>')


def parse_tool_call(text):
    s=text.strip()
    fence=chr(96)*3
    if s.startswith(fence):
        rest=s[len(fence):]
        nl=rest.find('\n')
        s=rest[nl+1:] if nl>=0 else rest
        s=s.rstrip()
        if s.endswith(fence):s=s[:-3]
        s=s.strip()
        if s.lower().startswith('json'):s=s[4:].strip()
    try:return json.loads(s)
    except json.JSONDecodeError:
        m=re.search(r'\{.*\}',s,re.DOTALL)
        if m:
            try:return json.loads(m.group())
            except json.JSONDecodeError:return None
        return None


def structural_check(parsed,subtask_name,required):
    if not isinstance(parsed,dict):return False,'invalid_json'
    if parsed.get('tool')!=subtask_name:return False,'wrong_tool'
    args=parsed.get('args')
    if not isinstance(args,dict):return False,'no_args'
    for f in required:
        if f not in args:return False,f'missing_key:{f}'
        v=args[f]
        if not isinstance(v,str) or not v.strip():return False,f'empty_value:{f}'
        if PLACEHOLDER_RE.search(v):return False,f'placeholder:{f}'
    return True,None

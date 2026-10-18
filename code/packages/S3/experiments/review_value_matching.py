"""Conservative source-value matching. This is lexical coverage, not semantic truth."""
import re
import unicodedata

MATCH_VERSION = 'source-lexical-numeric-boundaries-2'
MATCH_DESCRIPTION = ('Unicode width and whitespace normalized; punctuation, signs, decimals, '
                     'percentages, units and numeric expression boundaries preserved. '
                     'No unit/date conversion or semantic-equivalence inference; lexical coverage only.')
# Units remain part of an atomic numeric expression: 100 must not match 100万.
UNIT = r'(?:万美元|亿美元|人民币|万元|亿元|千元|美元|毫克|千克|公斤|厘米|毫米|公里|毫升|小时|分钟|万元|万|亿|千|百|元|岁|年|月|日|号|点|时|分|秒|天|晚|米|克|升|%|‰)'
NUMBER = r'[+\-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:[eE][+\-]?\d+)?'
NUMERIC = re.compile(NUMBER + r'(?:' + UNIT + r')*(?:(?:[-/:~至到])' + NUMBER + r'(?:' + UNIT + r')*)*')


def normalize_value(value):
    """Width and whitespace only; retain punctuation, signs, scale and date syntax."""
    text = unicodedata.normalize('NFKC', str(value))
    text = text.translate(str.maketrans({'−': '-', '﹣': '-'}))
    return ''.join(c for c in text if not c.isspace())


def source_value_match(value, argument):
    v, a = normalize_value(value), normalize_value(argument)
    if not v or not isinstance(argument, str):
        return False
    expected_numbers = [(m.start(), m.end(), m.group()) for m in NUMERIC.finditer(v)]
    actual_numbers = {(m.start(), m.end(), m.group()) for m in NUMERIC.finditer(a)}
    for occurrence in re.finditer(re.escape(v), a):
        start, end = occurrence.span()
        # ASCII identifiers cannot match just a prefix or suffix of another identifier.
        if v[0].isascii() and v[0].isalnum() and start and a[start-1].isascii() and a[start-1].isalnum():
            continue
        if v[-1].isascii() and v[-1].isalnum() and end < len(a) and a[end].isascii() and a[end].isalnum():
            continue
        # Every numeric atom must have exactly the same extent and spelling in context.
        if all((start+s, start+e, token) in actual_numbers for s,e,token in expected_numbers):
            return True
    return False

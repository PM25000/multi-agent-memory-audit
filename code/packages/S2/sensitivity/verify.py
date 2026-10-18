"""Replay sensitivity estimates from scored group contributions, without answers.

Usage: python verify_group_sensitivity_counts.py DIRECTORY
DIRECTORY contains contributions.json and summary.json. This does not validate
human semantic judgments, sampling coverage or the original response-to-count map.
"""
import argparse
import json
import math
from pathlib import Path


def verify(directory):
    rows=json.loads((directory/'contributions.json').read_text(encoding='utf-8'))
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert len(rows)==96*3*2 and not summary['confidence_intervals']
    seen=set();totals={}
    for row in rows:
        key=(row['group'],row['model'],row['endpoint']);assert key not in seen;seen.add(key)
        for scenario,bounds in row['counts'].items():
            acc=totals.setdefault((row['model'],row['endpoint'],scenario),[[0.,0.],[0.,0.]])
            assert 0 <= bounds[0][0] <= bounds[0][1] <= 3
            assert 0 <= bounds[1][0] <= bounds[1][1] <= bounds[0][1]
            for j in (0,1):
                for k in (0,1):acc[j][k]+=row['weight']*bounds[j][k]
    checked=0
    for (model,ep,scenario),(y,z) in totals.items():
        assert len([k for k in seen if k[1:]==(model,ep)])==96
        expected=summary['results'][model][ep][scenario]
        actual={'added_per_input':[v/150 for v in y],
                'correct_share':[z[0]/y[1] if y[1] else 0.,min(1.,z[1]/y[0]) if y[0] else 1.]}
        for field in actual:
            assert all(math.isclose(a,b,abs_tol=1e-9) for a,b in zip(actual[field],expected[field]))
        checked+=1
    assert checked==24
    return {'status':'PASS','scored_group_model_endpoints':len(rows),'estimate_rows_checked':checked,
            'human_semantics_validated':False,'new_model_calls':0,'sampling_confidence_claim':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(verify(parser.parse_args().directory),ensure_ascii=False,indent=2))

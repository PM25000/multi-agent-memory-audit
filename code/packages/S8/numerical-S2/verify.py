"""Independent numerical replay from group scores; Python standard library only."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import statistics


def verify():
    root = Path(__file__).resolve().parent
    read = lambda n: json.loads((root/n).read_text(encoding='utf-8'))
    for name, sha in read('manifest.json')['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha, name
    d, expected = read('groups.json'), read('expected.json')
    sizes = {int(k): v for k,v in d['sizes'].items()}
    ns = {int(k): v for k,v in d['sample_sizes'].items()}
    assert Counter(g['stratum'] for g in d['groups']) == ns
    assert len({g['group'] for g in d['groups']}) == 96
    radius = math.sqrt(math.log(480)/2*sum(sizes[h]**2*(h-1)**2/ns[h] for h in sizes))
    report = {}
    def close(a,b):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
    for model, eps in expected.items():
        report[model] = {}
        for ep, target in eps.items():
            values = {h: [g['counts'][model][ep] for g in d['groups'] if g['stratum']==h] for h in sizes}
            for h, rows in values.items():
                assert all(0 <= s <= n <= h-1 for n,s in rows)
            tn,ts = [sum(sizes[h]*statistics.mean(v[j] for v in values[h]) for h in sizes) for j in (0,1)]
            ratio = ts/tn
            close(tn/d['input_denominator'],target['point_estimate']['added_per_input'])
            close(ratio,target['point_estimate']['correct_share'])
            bounds = [[max(0,t-radius), min(d['capacities'][model],t+radius)] for t in (tn,ts)]
            for j,name in enumerate(('unnecessary','correct_spillover')):
                for a,b in zip(bounds[j],target['simultaneous_95_total_bounds'][name]): close(a,b)
            share = [bounds[1][0]/bounds[0][1],min(1,bounds[1][1]/bounds[0][0])]
            for a,b in zip(share,target['simultaneous_95_correct_share']):close(a,b)
            for a,b in zip(bounds[0],target['simultaneous_95_added_per_input']):close(a/150,b)
            vn = sum(sizes[h]**2*(1-ns[h]/sizes[h])*statistics.variance(v[0] for v in values[h])/ns[h] for h in sizes)
            vr = sum(sizes[h]**2*(1-ns[h]/sizes[h])*statistics.variance(v[1]-ratio*v[0] for v in values[h])/ns[h] for h in sizes)/tn**2
            approx_n = [max(0,tn-1.95996398454*math.sqrt(vn))/150,min(d['capacities'][model],tn+1.95996398454*math.sqrt(vn))/150]
            approx_r = [max(0,ratio-1.95996398454*math.sqrt(vr)),min(1,ratio+1.95996398454*math.sqrt(vr))]
            for key,actual in [('added_per_input',approx_n),('correct_share',approx_r)]:
                for a,b in zip(actual,target['approximate_design_95'][key]):close(a,b)
            report[model][ep] = {'added_per_input':tn/150,'correct_share':ratio,'simultaneous_added_bounds':[x/150 for x in bounds[0]],'simultaneous_share_bounds':share}
    from sensitivity.verify import verify as verify_sensitivity
    sensitivity = verify_sensitivity(root/'sensitivity')
    by_group = {g['group']:g for g in d['groups']}
    for row in read('sensitivity/contributions.json'):
        g=by_group[row['group']];h=g['stratum']
        close(row['weight'],sizes[h]/ns[h])
        expected_counts=g['counts'][row['model']][row['endpoint']]
        assert row['counts']['adjudicated']==[[v,v] for v in expected_counts]
    return {'status':'PASS','groups':96,'radius_per_input':radius/150,'results':report,
            'posthoc_adjudication_sensitivity':sensitivity,
            'same_group_weights_and_adjudicated_counts':True}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))

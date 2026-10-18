"""Standard-library replay of S5 saved selections; no network or model calls."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    d=read(ROOT/'dataset.json');expected=read(ROOT/'summary.json')
    totals={k:0 for k in expected['primary_totals']};human=dict(totals);count=0
    for w in d['workflows']:
        if w.get('analysis_stratum')!='primary_candidate':continue
        row=read(ROOT/'selection'/(w['id'].replace('/','__')+'.json'))
        if row['selected'] is None:continue
        count+=1;sets={k:set(v) for k,v in row['selected'].items()};u=set().union(*sets.values())
        refs={r['id']:set(r['required_ids']) for r in w['receivers']}
        for role in w['receivers']:
            i=role['id'];added=u-sets[i];bad=added-refs[i]
            good={e for e in bad if any(e in sets[j]&refs[j] for j in sets)}
            metrics={'D_nonrequired':len(sets[i]-refs[i]),'U_nonrequired':len(u-refs[i]),
                     'added_nonrequired':len(bad),'correct_donor_spillover':len(good),
                     'error_only_spillover':len(bad-good),'added_required':len(added&refs[i])}
            for k,v in metrics.items():
                totals[k]+=v
                if role['reference_provenance']['human_checked']:human[k]+=v
    assert count==expected['observed_primary_workflows']==11
    assert totals==expected['primary_totals']
    assert human==expected['human_checked_recipient_totals']
    print(json.dumps({'verified':True,'workflows':count,'totals':totals,'human_checked_subset':human}))
if __name__=='__main__':main()

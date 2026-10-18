"""Read-only verification of added agreement and worked-group records; no network."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,runpy,sys,zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
read=lambda n:json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
def verify(s3_zip=None):
 base=runpy.run_path(str(ROOT/'verify.py')); result=base['verify']()
 sheets={k:read(f'evidence/judgments_{k}.json')['answers'] for k in ('A','B','C')}
 summary=read('evidence/AB-agreement-summary.json'); categories=['not_required','possible_only','required']; matrix=[[0]*3 for _ in categories]
 for gid,ans in sheets['A'].items():
  for task,value in ans['needs'].items():matrix[categories.index(value['status'])][categories.index(sheets['B'][gid]['needs'][task]['status'])]+=1
 rows=[sum(row) for row in matrix]; cols=[sum(row[j] for row in matrix) for j in range(3)];n=sum(rows);agree=sum(matrix[i][i] for i in range(3));pe=sum(x*y for x,y in zip(rows,cols))/n**2;kappa=(agree/n-pe)/(1-pe)
 assert summary['categories']==categories and summary['confusion_matrix']==matrix
 assert summary['A_marginal']==rows and summary['B_marginal']==cols and summary['judgments']==n and summary['agreements']==agree
 assert math.isclose(summary['raw_agreement'],agree/n) and math.isclose(summary['expected_agreement_from_marginals'],pe) and math.isclose(summary['unweighted_nominal_cohen_kappa'],kappa)
 for name,sha in summary['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 example=read('evidence/worked-example.json');gid=example['group_id'];sample=read('evidence/sample.json');g=next(g for g in sample if g['group_id']==gid);resolved=read('evidence/resolved.json')[gid]
 assert example['case_key']==g['case_key'] and example['entity_id']==g['entity_id']
 model=example['model'];rec=read('evidence/retained-grades.json')[model][g['case_key']];index=rec['memory_entity_ids'].index(g['entity_id'])
 assert example['grade_by_task']=={str(t['task_index']):t['grades'][index] for t in rec['tasks']}
 assert example['selectors']==g['selectors'][model] and example['added_recipients']==g['added'][model]
 for k in ('A','B'):assert example[f'{k}_necessity']=={t:v['status'] for t,v in sheets[k][gid]['needs'].items()}
 assert example['C_present']==(gid in sheets['C']) and math.isclose(example['design_weight'],g['design_weight'])
 for endpoint,counts in example['endpoint_counts'].items():
  actual=base['outcome'](g,resolved,model,endpoint)
  assert counts=={'non_required_added':actual[0][0],'correct_selection_spillover':actual[1][0]}
  assert all(lo==hi for lo,hi in actual)
 linkmap=read('evidence/S3-selection-link-map.json'); grades=read('evidence/retained-grades.json');links=linkmap['links'];assert len(links)==450
 assert {(x['model'],x['case_key']) for x in links}=={(m,k) for m,rs in grades.items() for k in rs}
 for link in links:
  source=grades[link['model']][link['case_key']];assert source['source_sha256']==link['source_sha256']
  assert link['S3_zip_member']==source['source_path'] or link['S3_zip_member'].endswith('/'+source['source_path'])
 if s3_zip:
  zp=Path(s3_zip);assert hashlib.sha256(zp.read_bytes()).hexdigest()==linkmap['S3_zip_sha256']
  with zipfile.ZipFile(zp) as z:
   for link in links:assert hashlib.sha256(z.read(link['S3_zip_member'])).hexdigest()==link['source_sha256']
 cases=read('evidence/cases.json');han=lambda text:any('\u4e00'<=c<='\u9fff' for c in text)
 assert all(han(c['item']['user_input']) and han(c['item']['memory_context']) for c in cases)
 result.update({'reading_aid_records_verified':True,'original_AB_matrix':matrix,'unweighted_nominal_kappa':kappa,'worked_group':gid,'S3_record_links_verified':len(links),'S3_original_zip_bytes_checked':bool(s3_zip),'core_cases_with_Chinese_request_and_memory':len(cases),'reading_aid_boundary':'Record identities and derived counts checked; English translations are inspection aids, not independent semantic validation.'})
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--s3-zip');args=parser.parse_args();print(json.dumps(verify(args.s3_zip),ensure_ascii=False,indent=2))

"""Recompute domain MPJPE and the paper notebook's aggregation from predictions."""
from pathlib import Path
import pickle,json,argparse
import numpy as np

def stats(path):
    obj=pickle.load(path.open('rb'))
    errors=[]
    for pred,label in zip(obj['predicts'],obj['labels']):
        pred=np.asarray(pred).reshape(-1,14,2);label=np.asarray(label).reshape(-1,14,2)
        assert pred.shape==label.shape and np.isfinite(pred).all()
        errors.extend(np.linalg.norm(pred-label,axis=2).mean(axis=1).tolist())
    return dict(samples=len(errors),mean=float(np.mean(errors)),population_sd=float(np.std(errors)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,default=Path(__file__).resolve().parents[1]/'local/hupr-20261008');a=p.parse_args()
    rows=[]
    for arm in ['released','ssr','crate']:
        run=a.campaign/arm
        if not run.exists():continue
        if not (run/'complete.json').exists():print(arm,'not complete');continue
        files=list((run/'results').glob('*.pkl'))
        ind=[f for f in files if '_indomain_test' in f.name];out=[f for f in files if '_indomain_test' not in f.name]
        assert len(ind)==len(out)==1,(arm,files)
        i,o=stats(ind[0]),stats(out[0]);row=dict(arm=arm,in_domain=i,cross_domain=o,
            paper_notebook_mean=(i['mean']+o['mean'])/2,paper_notebook_spread=(i['population_sd']+o['population_sd'])/2)
        rows.append(row);print(json.dumps(row))
    (a.campaign/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()

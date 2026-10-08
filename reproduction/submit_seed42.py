"""Submit the prepared independent RF-CRATE seed-42 pair under the invoking user."""
from pathlib import Path
import subprocess,json,os
ROOT=Path(__file__).resolve().parents[1]
camp=ROOT/'local/hupr-seed42-20261008'
os.umask(0)
try:
    (camp/'submission.lock').mkdir()
except FileExistsError:
    receipt=camp/'submission.json'
    records=json.loads(receipt.read_text()) if receipt.exists() else []
    print('This campaign is already locked for submission; no duplicate jobs were submitted.')
    for record in records:print(record['arm'],record['job'])
    if len(records)!=2:print('Submission may be partial; inspect the receipt before retrying.')
    raise SystemExit(0 if len(records)==2 else 1)
records=[]
for arm in ('released','ssr'):
    cmd=['sbatch','--parsable','--job-name=rfcrate-seed42-'+arm,'--time=03:00:00',
         '--output='+str(camp/f'{arm}-%j.out'),str(camp/'job.sbatch'),'--arm',arm]
    job=subprocess.check_output(cmd,text=True).strip()
    records.append(dict(arm=arm,job=job,command=cmd))
    (camp/'submission.json').write_text(json.dumps(records,indent=2)+'\n')
    print(arm,job,flush=True)

"""Prepare a new isolated RF-CRATE HuPR campaign without submitting it."""
import argparse,subprocess,shutil,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--radar-maps',type=Path,required=True)
    p.add_argument('--file-order',type=Path,help='Recorded basename list; otherwise capture current directory order')
    p.add_argument('--seed',type=int,help='Override each released configuration seed')
    p.add_argument('--cache',type=Path,help='Reuse an already completed cache instead of converting it again')
    a=p.parse_args();camp=a.campaign.resolve();camp.mkdir(parents=True,exist_ok=False);src=camp/'source';src.mkdir()
    files=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z'],text=True).split('\0')
    for f in files:
        if f and (f.endswith('.py') or f.endswith('.yaml')) and not f.startswith('reproduction/'):
            dest=src/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    subprocess.run(['patch','-p1','-i',str(ROOT/'reproduction/source-changes.patch')],cwd=src,check=True)
    cache=camp/'cache'
    if a.cache:
        shared=a.cache.resolve()
        order=json.loads((shared/'file_order.json').read_text())
        for name in order:
            assert (shared/Path(name).stem/'complete.json').exists(), name
        if a.file_order:
            assert [Path(name).name for name in order]==json.loads(a.file_order.read_text())
        cache.symlink_to(shared,target_is_directory=True)
    else:
        cache.mkdir()
        files={p.name:p.resolve() for p in a.radar_maps.iterdir() if p.suffix=='.pkl'}
        names=json.loads(a.file_order.read_text()) if a.file_order else list(files)
        assert len(names)==len(set(names)) and set(names)==set(files), 'Recorded order must cover the same files exactly'
        order=[str(files[name]) for name in names]
        (cache/'file_order.json').write_text(json.dumps(order,indent=2)+'\n')
    for arm,base in [('released','HuPR_rfcrate_mini'),('ssr','HuPR_rfcrate_mini'),('crate','HuPR_CRATEsmall')]:
        cfg=yaml.safe_load((src/'Configurations/HuPR'/f'{base}.yaml').read_text())
        cfg.update(task='reproduction_'+arm,dataset_path=str(cache),num_workers=8,model_save_enable=True,tqdm_disable=True)
        if a.seed is not None:cfg['init_rand_seed']=a.seed
        if arm in ('released','ssr'):cfg['ssr']=arm=='ssr'
        for key,d in [('tensorboard_folder','tensorboard'),('trained_model_folder','weights'),('log_folder','results')]:cfg[key]=str(camp/arm/d)+'/'
        (src/'Configurations/HuPR'/f'reproduction_{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(camp)
if __name__=='__main__':main()

"""Verify completed runs and export compact evidence, leaving large outputs local."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = json.loads((args.campaign/'summary.json').read_text())
    assert rows, 'Run summarize_hupr.py after training first'
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        arm = row['arm']
        run = args.campaign/arm
        assert (run/'complete.json').exists(), arm
        dest = args.output/arm
        dest.mkdir(exist_ok=True)
        cfg_path = args.campaign/'source/Configurations/HuPR'/f'reproduction_{arm}.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        latest = torch.load(run/'weights/latest.pth', map_location='cpu')
        best = torch.load(run/'weights/best.pth', map_location='cpu')
        named = list((run/'weights').glob('reproduction_*.pth'))
        assert len(named) == 1, named
        evaluated = torch.load(named[0], map_location='cpu')
        for candidate in [latest['best_model'], evaluated]:
            assert best.keys() == candidate.keys()
            assert all(torch.equal(best[key], candidate[key]) for key in best)
        logs = [p for p in args.campaign.glob('*.out')
                if p.name.startswith(f'train-{arm}-') or p.name.startswith(f'{arm}-')]
        assert len(logs) == 1, logs
        log = logs[0].read_text()
        epochs = [int(value)+1 for value in re.findall(r'Epoch (\d+), Average Loss \(train set\)', log)]
        assert epochs
        row.update(seed=cfg['init_rand_seed'], trained_epochs=max(epochs),
                   selected_epoch=latest['best_epoch']+1,
                   last_saved_resume_epoch=latest['epoch'],
                   slurm_job=json.loads((run/'started.json').read_text())['slurm_job'],
                   selected_weights_verified=True,
                   best_checkpoint_sha256=sha256(run/'weights/best.pth'),
                   prediction_sha256={p.name:sha256(p) for p in sorted((run/'results').glob('*.pkl'))})
        shutil.copy2(cfg_path, dest/'configuration.yaml')
        shutil.copy2(run/'weights/history.json', dest/'history.json')
        shutil.copy2(logs[0], dest/'run.log')
        (dest/'result.json').write_text(json.dumps(row, indent=2)+'\n')
        print(arm, 'seed', row['seed'], 'selected epoch', row['selected_epoch'],
              'trained epochs', row['trained_epochs'], 'weights verified')
    (args.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()

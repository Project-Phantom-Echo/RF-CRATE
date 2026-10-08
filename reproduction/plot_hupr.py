"""Plot saved training histories without loading models or touching checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = dict(released='#4677bb', ssr='#21936d', crate='#b76a34')
    for campaign_index, campaign in enumerate(args.campaign):
        for arm, color in colors.items():
            path = campaign/arm/'weights/history.json'
            if not path.exists():
                continue
            history = json.loads(path.read_text())
            if not history:
                continue
            config = yaml.safe_load((campaign/'source/Configurations/HuPR'/f'reproduction_{arm}.yaml').read_text())
            names = dict(released='RF-CRATE, SSR off', ssr='RF-CRATE, SSR on', crate='CRATE')
            label = f'{names[arm]}, seed {config["init_rand_seed"]}'
            style = '-' if campaign_index == 0 else '--'
            epochs = [row['epoch'] for row in history]
            for ax, key in zip(axes, ['train_loss', 'validation_mpjpe']):
                ax.plot(epochs, [row[key] for row in history], style,
                        color=color, label=label, linewidth=1.5)
    for ax in axes:
        ax.set_xlabel('Completed epoch (1-based)')
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Training task loss (log scale)')
    axes[1].set_ylabel('Validation MPJPE, pixels (log scale)')
    axes[1].legend(fontsize=7)
    fig.suptitle('RF-CRATE HuPR runs: released schedule and validation selection')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == '__main__':
    main()

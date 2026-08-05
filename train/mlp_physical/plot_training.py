"""
Plot the training curves written by train.py.

Reads `train_log.csv` and `valid_log.csv` from a checkpoint directory and emits
a single figure. Safe to run while training is still going — it just plots
whatever has been written so far.

The per-parameter panel is the one to read. `rms_<param>` is the RMS error on
that alignment parameter, and at initialisation it sits at the width of the
training distribution, because a model that has learned nothing predicts zero.
The dashed line marks that starting level: a curve that stays on it means the
parameter is not being learned, regardless of what the total loss does.

Usage
-----
    python train/mlp_physical/plot_training.py \\
        --checkpoint-dir train/mlp_physical/checkpoints_spread_vertex

    # compare several runs on the same axes
    python train/mlp_physical/plot_training.py \\
        --checkpoint-dir run_a run_b --labels baseline constrained
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_PARAMS = ['dx', 'dy', 'dz', 'nu', 'nv', 'rho']
_PARAM_COLORS = ['steelblue', 'seagreen', 'darkorange',
                 'firebrick', 'mediumpurple', 'saddlebrown']


def read_log(path):
    """Read a training log into {column: np.ndarray}. Returns None if absent."""
    path = Path(path)
    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    out = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(np.nan)
        out[key] = np.array(values)
    return out


def check_continuity(log, label):
    """
    Warn when a log holds more than one run.

    train.py appends, so deleting a checkpoint without deleting the log leaves
    epochs like 1 2 1 2 in one file. Plotting that as a single curve is
    misleading, so say so rather than quietly drawing it.
    """
    epoch = log.get('epoch')
    if epoch is None or len(epoch) < 2:
        return
    restarts = int(np.sum(np.diff(epoch) <= 0))
    if restarts:
        print(f'WARNING: {label} contains {restarts + 1} runs appended into one '
              f'file (epoch resets {restarts} time(s)). The curve mixes them; '
              f'delete the whole checkpoint directory between runs, not just '
              f'the checkpoint.')


def _plot_pair(ax, train, valid, key, title, ylabel, logy=True):
    if train is not None and key in train:
        ax.plot(train['epoch'], train[key], color='steelblue',
                linewidth=1.4, label='train')
    if valid is not None and key in valid:
        ax.plot(valid['epoch'], valid[key], color='darkorange',
                linewidth=1.4, label='valid')
    if logy:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('epoch', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def plot_run(train, valid, output, title):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(title, fontsize=12)

    _plot_pair(axes[0, 0], train, valid, 'loss',
               'Total loss', 'param + geom_weight x geom')
    _plot_pair(axes[0, 1], train, valid, 'param',
               'Parameter term (weighted MSE)', 'param')
    _plot_pair(axes[0, 2], train, valid, 'geom',
               'Geometric term (hit displacement)', 'mm^2')

    # per-parameter RMS, with the "predicts zero" starting level marked
    ax = axes[1, 0]
    source = valid if valid is not None else train
    for j, name in enumerate(_PARAMS):
        key = f'rms_{name}'
        if source is None or key not in source:
            continue
        series = source[key]
        ax.plot(source['epoch'], series, color=_PARAM_COLORS[j],
                linewidth=1.4, label=name)
        ax.axhline(series[0], color=_PARAM_COLORS[j], linewidth=0.7,
                   linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    ax.set_title('Per-parameter RMS error (valid)\n'
                 'dotted = level at epoch 1, i.e. predicting zero', fontsize=9)
    ax.set_xlabel('epoch', fontsize=8)
    ax.set_ylabel('RMS error', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    # fractional improvement over the epoch-1 level — the clearest "is it
    # learning this parameter at all" view
    ax = axes[1, 1]
    for j, name in enumerate(_PARAMS):
        key = f'rms_{name}'
        if source is None or key not in source:
            continue
        series = source[key]
        if series[0] > 0:
            ax.plot(source['epoch'], series / series[0],
                    color=_PARAM_COLORS[j], linewidth=1.4, label=name)
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_ylim(0, 1.15)
    ax.set_title('RMS relative to epoch 1\n1.0 = not learned', fontsize=9)
    ax.set_xlabel('epoch', fontsize=8)
    ax.set_ylabel('rms / rms(epoch 1)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 2]
    if train is not None and 'lr' in train:
        ax.plot(train['epoch'], train['lr'], color='k', linewidth=1.4)
    ax.set_yscale('log')
    ax.set_title('Learning rate', fontsize=10)
    ax.set_xlabel('epoch', fontsize=8)
    ax.set_ylabel('lr', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'saved {output}')


def plot_comparison(runs, output):
    """One panel per quantity, one line per run."""
    keys = [('loss', 'Total loss', True),
            ('geom', 'Geometric term (mm^2)', True),
            ('rms_rho', 'RMS error: rho', True),
            ('rms_nu', 'RMS error: nu', True),
            ('rms_nv', 'RMS error: nv', True),
            ('rms_dy', 'RMS error: dy', True)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle('Run comparison (validation)', fontsize=12)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 2)))
    for ax, (key, title, logy) in zip(axes.ravel(), keys):
        for (label, _, valid), color in zip(runs, colors):
            source = valid
            if source is None or key not in source:
                continue
            ax.plot(source['epoch'], source[key], linewidth=1.4,
                    color=color, label=label)
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('epoch', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'saved {output}')


def get_args():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--checkpoint-dir', type=str, nargs='+', required=True,
                        help='one or more checkpoint directories')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='names for the runs; defaults to directory names')
    parser.add_argument('--output', type=str, default=None,
                        help='output PNG; defaults to training_curves.png inside '
                             'the first checkpoint directory')
    return parser.parse_args()


def main():
    args = get_args()

    labels = args.labels or [Path(d).name for d in args.checkpoint_dir]
    if len(labels) != len(args.checkpoint_dir):
        raise ValueError('--labels must have one entry per --checkpoint-dir')

    runs = []
    for label, directory in zip(labels, args.checkpoint_dir):
        train = read_log(Path(directory) / 'train_log.csv')
        valid = read_log(Path(directory) / 'valid_log.csv')
        if train is None and valid is None:
            raise FileNotFoundError(f'no train_log.csv or valid_log.csv in {directory}')
        for log, name in ((train, f'{label}/train_log.csv'),
                          (valid, f'{label}/valid_log.csv')):
            if log is not None:
                check_continuity(log, name)
        n = len(train['epoch']) if train is not None else len(valid['epoch'])
        print(f'{label}: {n} epochs')
        runs.append((label, train, valid))

    output = Path(args.output) if args.output else \
        Path(args.checkpoint_dir[0]) / 'training_curves.png'
    output.parent.mkdir(parents=True, exist_ok=True)

    if len(runs) == 1:
        label, train, valid = runs[0]
        plot_run(train, valid, output, f'Training curves — {label}')
    else:
        plot_comparison(runs, output)


if __name__ == '__main__':
    main()

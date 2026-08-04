"""
Build a train/test dataset in the layout ROMDataset expects.

    <output-root>/train/sample_*.npz
    <output-root>/test/sample_*.npz

The two splits are given disjoint seed ranges so they share no events.

Usage
-----
    python -m rtal.data.make_dataset \\
        --config      train/mlp_physical/dataset.yaml \\
        --output-root data/rom_det-3_part-200 \\
        --num-train   8000 --num-test 1000

Then point the training script at it:

    export DATAROOT=$PWD/data/rom_det-3_part-200
"""
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from rtal.data.generate import generate_one

# Seed ranges must not overlap between splits, or events leak from train to
# test.  One million is far more than any split we would generate.
_SPLIT_SEED_BASE = {'train': 0, 'test': 1_000_000}


def make_split(config, output_root, split, num_samples, overwrite=False):
    """Generate one split.  Returns the directory written."""
    out_dir = Path(output_root) / split
    existing = sorted(out_dir.glob('*.npz')) if out_dir.exists() else []

    if existing and not overwrite:
        raise ValueError(
            f'{out_dir} already holds {len(existing)} samples. '
            f'Pass --overwrite to regenerate, or choose another --output-root.'
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in existing:
        sample.unlink()

    base_seed = _SPLIT_SEED_BASE[split]
    for idx in tqdm(range(num_samples), desc=f'{split:>5}'):
        generate_one(config, out_dir, f'sample_{idx}', seed=base_seed + idx)

    return out_dir


def get_args():
    parser = argparse.ArgumentParser(description='Generate a train/test dataset')
    parser.add_argument('--config',      type=str, required=True,
                        help='data generation config yaml')
    parser.add_argument('--output-root', type=str, required=True,
                        help='root directory; train/ and test/ are created under it')
    parser.add_argument('--num-train',   type=int, default=8000)
    parser.add_argument('--num-test',    type=int, default=1000)
    parser.add_argument('--overwrite',   action='store_true',
                        help='delete and regenerate splits that already exist')
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    print(f'config          : {args.config}')
    print(f'num_particles   : {config["dataset"]["num_particles"]}')
    print(f'rounded readout : {config["dataset"]["rounded"]}')
    print(f'detectors       : {len(config["detectors"])}')
    print(f'misalignments   : '
          f'{[k for k in config["misalignments"] if k != "max_num_misalignments"]}\n')

    for split, count in (('train', args.num_train), ('test', args.num_test)):
        out_dir = make_split(config, args.output_root, split, count,
                             overwrite=args.overwrite)
        print(f'wrote {count} samples to {out_dir}')

    print(f'\nDone. Now run:\n    export DATAROOT={Path(args.output_root).resolve()}')


if __name__ == '__main__':
    main()

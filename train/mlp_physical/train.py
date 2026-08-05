"""
Train the physical-parameter alignment model.

Differences from train/mlp_no-residual/train.py:

  * The model predicts (dx, dy, dz, nu, nv, rho) per detector instead of the 9
    raw detector numbers, so roll is no longer entangled with tilt.
  * The readout is normalised on the way in (inside the model).
  * The loss has two terms:

      param — MSE on the six parameters, each weighted by how far a unit of it
              actually moves the hits.  A plain MSE implicitly weights by the
              spread of the target, which on this distribution hands ~77% of
              the budget to the orientation block and, within it, mostly to the
              near-unobservable tilt.
      geom  — mean squared hit displacement (mm^2) after correcting with the
              predicted geometry.  This is the quantity the diagnostics plot,
              and it weights every degree of freedom by its observability
              automatically.

    The original script computed a residual term and then multiplied it by zero
    (`loss = diff + 0 * residual`); here it is a real, configurable term.
  * Multi-GPU via DistributedDataParallel.

Running
-------
Single GPU:

    python train.py --config config.yaml --device cuda --gpu-id 0

All 8 GPUs on one node:

    torchrun --standalone --nproc_per_node=8 train.py --config config.yaml

`batch_size` in the config is **per GPU**, so the effective batch is
batch_size * nproc_per_node.  With `train.scale_lr: true` the learning rate is
multiplied by the world size to match (linear scaling rule).

Only rank 0 writes checkpoints, logs and progress bars.  Metrics are reduced
across ranks first, so the logged numbers are whole-dataset values.
"""

import os
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from rtal.datasets.dataset import ROMDataset
from rtal.utils import get_lr, count_parameters, get_data_root
from rtal.utils import Checkpointer
from rtal.models.mlp_physical import (PhysicalMLP,
                                      params_to_detector,
                                      detector_to_params)
from rtal.geometry.line import reconstruct, get_center_basis

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

PITCH = 0.1

# Scalar terms accumulated per batch, then reduced across ranks.  The six
# per-parameter entries are mean squared errors; the reported rms_* values are
# their square roots, taken after reduction so they are true whole-dataset RMS.
_SCALAR_KEYS = ['loss', 'param', 'geom', 'mse_det']
_PARAM_KEYS  = ['dx', 'dy', 'dz', 'nu', 'nv', 'rho']


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """
    Initialise the process group when launched under torchrun.

    Returns (is_distributed, rank, world_size, local_rank).  Falls back to
    single-process when the torchrun environment variables are absent, so the
    same script still runs as `python train.py`.
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, 0, 1, 0

    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)

    return True, rank, world_size, local_rank


def reduce_totals(totals, count, is_distributed):
    """Sum per-rank accumulators so every rank logs whole-dataset numbers."""
    if not is_distributed:
        return totals, count

    packed = torch.cat([totals, count.reshape(1)])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return packed[:-1], packed[-1]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def local_residual(detector_pred, detector_start, readout_curr, readout_start):
    """
    Mean squared hit displacement (mm^2) in the nominal local frame.

    Reconstructs each hit with the predicted geometry, projects it into the
    start frame, and compares with where the hit would have landed had the
    detector never moved.  Identical to the "local after" residual reported by
    simulate_sliding_window.py, so training and diagnostics measure the same
    thing.
    """
    points = reconstruct(detector_pred, readout_curr)             # (B, P, D, 3)
    center, basis = get_center_basis(detector_start)              # (B, D, 3), (B, D, 2, 3)

    displacement = points - center.unsqueeze(1)                   # (B, P, D, 3)
    local_xy = torch.einsum('bpdr,bdsr->bpds', displacement, basis)

    return ((local_xy - readout_start * PITCH) ** 2).mean()


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(model,
              core,
              rounded_readout,
              param_weights,
              geom_weight,
              dataloader, *,
              device,
              is_distributed,
              is_main,
              optimizer=None,
              desc=None):
    """
    Run one epoch over a data loader.

    `core` is the unwrapped model — DDP only proxies forward(), so anything
    else (the misalignment layer, inference()) must go through the module
    itself.
    """
    readout_type = 'rounded' if rounded_readout else 'cont'
    mse = nn.MSELoss()

    totals = torch.zeros(len(_SCALAR_KEYS) + len(_PARAM_KEYS), device=device)
    count  = torch.zeros((), device=device)

    pbar = tqdm(dataloader, total=len(dataloader), desc=desc, disable=not is_main)
    for event in pbar:

        # (B, D, P, 2) -> (B, P, D, 2)
        readout_curr  = torch.transpose(event[f'readout_curr_{readout_type}'], 1, 2).to(device, non_blocking=True)
        readout_start = torch.transpose(event[f'readout_start_{readout_type}'], 1, 2).to(device, non_blocking=True)

        detector_curr  = event['detector_curr'].to(device, non_blocking=True)
        detector_start = event['detector_start'].to(device, non_blocking=True)

        is_train = optimizer is not None

        model_input = readout_curr.flatten(-2, -1)                # (B, P, D*2)
        if is_train:
            params_pred = model(model_input)
        else:
            params_pred = core.inference(model_input, randperm=False)

        params_true   = detector_to_params(detector_curr, detector_start,
                                           core.misalign_layer)
        detector_pred = params_to_detector(params_pred, detector_start,
                                           core.misalign_layer)

        # weighted parameter loss — weights are "bins of hit displacement per
        # unit parameter", so every degree of freedom enters in the same units
        param_loss = (((params_pred - params_true) * param_weights) ** 2).mean()
        geom_loss  = local_residual(detector_pred, detector_start,
                                    readout_curr, readout_start)

        loss = param_loss + geom_weight * geom_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            per_param = (params_pred - params_true).pow(2).mean(dim=(0, 1))   # (6,)
            batch = torch.stack([loss.detach(),
                                 param_loss.detach(),
                                 geom_loss.detach(),
                                 mse(detector_pred, detector_curr).detach(),
                                 *per_param])
            totals += batch
            count  += 1

            if is_main:
                pbar.set_postfix({
                    'loss':    f'{(totals[0] / count).item():.6g}',
                    'geom':    f'{(totals[2] / count).item():.6g}',
                    'rms_rho': f'{(totals[-1] / count).sqrt().item():.6g}',
                    'rms_nu':  f'{(totals[-3] / count).sqrt().item():.6g}',
                })

    totals, count = reduce_totals(totals, count, is_distributed)
    means = (totals / count).cpu()

    summary = {key: means[i].item() for i, key in enumerate(_SCALAR_KEYS)}
    for j, key in enumerate(_PARAM_KEYS):
        summary[f'rms_{key}'] = means[len(_SCALAR_KEYS) + j].sqrt().item()

    return summary


# ---------------------------------------------------------------------------
# CLI / setup
# ---------------------------------------------------------------------------

def get_parameters():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='path to config file | default: config.yaml')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'),
                        help='ignored under torchrun, which always uses cuda')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='single-process runs only; torchrun sets this itself')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='dataloader workers per process | default: 8')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='override train.num_epochs; use this to extend a '
                             'finished run rather than editing the config, so the '
                             'config keeps recording what produced the checkpoint')
    parser.add_argument('--reschedule', action='store_true',
                        help='rebuild the LR schedule from the resume point. '
                             'Needed to change the decay of a resumed run at all: '
                             'the scheduler state, milestones included, is restored '
                             'from the checkpoint, so config edits are ignored')
    parser.add_argument('--lr', type=float, default=None,
                        help='with --reschedule, the rate to restart the decay '
                             'from; defaults to whatever the checkpoint is at')
    parser.add_argument('--sched-steps', type=int, default=None,
                        help='with --reschedule, epochs between LR decays')
    parser.add_argument('--sched-gamma', type=float, default=None,
                        help='with --reschedule, the decay factor')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    return config, args


def train():
    config, args = get_parameters()

    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    elif is_distributed:
        device = 'cpu'          # gloo fallback, used for testing the DDP path
    elif args.device == 'cuda':
        torch.cuda.set_device(args.gpu_id)
        device = f'cuda:{args.gpu_id}'
    else:
        device = 'cpu'

    config_dir = Path(args.config).resolve().parent

    checkpoint_path = Path(config['checkpointing']['checkpoint_path'])
    if not checkpoint_path.is_absolute():
        checkpoint_path = config_dir / checkpoint_path
    save_frequency = config['checkpointing']['save_frequency']
    resume         = config['checkpointing']['resume']

    num_epochs        = args.num_epochs or config['train']['num_epochs']
    num_warmup_epochs = config['train']['num_warmup_epochs']
    batch_size        = config['train']['batch_size']
    learning_rate     = config['train']['learning_rate']
    sched_steps       = config['train']['sched_steps']
    sched_gamma       = config['train']['sched_gamma']

    # linear scaling rule: the effective batch grew by world_size
    if config['train'].get('scale_lr', False):
        learning_rate *= world_size

    param_weights = torch.tensor(config['loss']['param_weights'],
                                 dtype=torch.float32, device=device)
    geom_weight   = config['loss']['geom_weight']

    if is_main:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()

    core      = PhysicalMLP(**config['model']).to(device)
    optimizer = AdamW(core.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer,
                            milestones=range(num_warmup_epochs, num_epochs, sched_steps),
                            gamma=sched_gamma)

    # The checkpointer holds the unwrapped model, so state dict keys carry no
    # 'module.' prefix and the checkpoint stays loadable by the single-process
    # diagnostics and the ONNX export path.
    checkpointer = Checkpointer(core,
                                optimizer       = optimizer,
                                scheduler       = scheduler,
                                checkpoint_path = checkpoint_path,
                                save_frequency  = save_frequency)
    resume_epoch = checkpointer.load(device=device) if resume else 0

    if resume_epoch and is_main:
        # The optimizer and scheduler states come from the checkpoint, and
        # MultiStepLR stores its milestones there too. So learning_rate,
        # sched_gamma, sched_steps and num_warmup_epochs in the config have no
        # effect when resuming — only num_epochs does. Say so rather than let it
        # look as though a config edit took hold.
        print(f'resuming at epoch {resume_epoch}; learning rate '
              f'{get_lr(optimizer):.6g} comes from the checkpoint, not the config')
        remaining = [m for m in getattr(scheduler, 'milestones', {})
                     if m > resume_epoch]
        if remaining:
            print(f'  {len(remaining)} LR milestone(s) remain: {sorted(remaining)}')
        else:
            print('  no LR milestones remain — the rate stays constant from here')

    if args.reschedule:
        if not resume_epoch:
            raise ValueError('--reschedule only applies when resuming a checkpoint')

        start_lr = args.lr if args.lr is not None else get_lr(optimizer)
        steps    = args.sched_steps or sched_steps
        gamma    = args.sched_gamma if args.sched_gamma is not None else sched_gamma

        for group in optimizer.param_groups:
            group['lr'] = start_lr
            # drop the stale value or the new scheduler inherits the original
            # base rate rather than the one we are restarting from
            group.pop('initial_lr', None)

        # Milestones are counted from the new scheduler's own epoch zero, which
        # is the resume point — not in absolute epochs.
        span = max(num_epochs - resume_epoch, 1)
        scheduler = MultiStepLR(optimizer,
                                milestones=range(steps, span + 1, steps),
                                gamma=gamma)
        checkpointer.scheduler = scheduler

        if is_main:
            # The rate used *during* the final epoch has seen only the decays
            # that fired strictly before it — the last step() lands after the
            # loop ends and never applies.
            n_decays = len([m for m in range(steps, span + 1, steps) if m < span])
            print(f'  rescheduled: {start_lr:.6g} decaying by {gamma} every {steps} '
                  f'epochs for the remaining {span}, '
                  f'reaching {start_lr * gamma ** n_decays:.6g} at epoch {num_epochs}')

    if is_distributed:
        # device_ids must be None for a CPU model (gloo), set for CUDA (nccl)
        model = DDP(core, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        model = core

    data_root       = get_data_root()
    num_particles   = config['data']['num_particles']
    rounded_readout = config['data']['rounded']

    train_ds = ROMDataset(data_root, split='train', num_particles=num_particles)
    valid_ds = ROMDataset(data_root, split='test',  num_particles=num_particles)

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False)
    else:
        train_sampler = valid_sampler = None

    loader_kwargs = {
        'batch_size':  batch_size,
        'num_workers': args.num_workers,
        'pin_memory':  device != 'cpu',
        'persistent_workers': args.num_workers > 0,
    }
    train_ldr = DataLoader(train_ds, sampler=train_sampler,
                           shuffle=train_sampler is None, **loader_kwargs)
    valid_ldr = DataLoader(valid_ds, sampler=valid_sampler,
                           shuffle=False, **loader_kwargs)

    if is_main:
        model_size = count_parameters(core) / (1024 ** 2)
        print(f'model size    : {model_size:.4f}MB')
        with open(checkpoint_path / 'model_size.dat', 'w', encoding='utf-8') as handle:
            handle.write(f'model_size_MB,{model_size:.2f}')

        print(f'world size    : {world_size}')
        print(f'batch size    : {batch_size} per GPU '
              f'(effective {batch_size * world_size})')
        print(f'learning rate : {learning_rate:g}'
              f'{" (scaled by world size)" if config["train"].get("scale_lr") else ""}')
        print(f'train samples : {len(train_ds)}, valid samples: {len(valid_ds)}')
        print(f'param weights : {config["loss"]["param_weights"]}, '
              f'geom weight: {geom_weight}\n')

    train_log = checkpoint_path / 'train_log.csv'
    valid_log = checkpoint_path / 'valid_log.csv'

    for epoch in range(resume_epoch + 1, num_epochs + 1):

        # reshuffles the per-rank split each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_lr = get_lr(optimizer)
        if is_main:
            print(f'current learning rate = {current_lr:.10f}')

        model.train()
        train_stat = run_epoch(model, core, rounded_readout,
                               param_weights, geom_weight, train_ldr,
                               desc           = f'Train Epoch {epoch} / {num_epochs}',
                               optimizer      = optimizer,
                               device         = device,
                               is_distributed = is_distributed,
                               is_main        = is_main)

        model.eval()
        with torch.no_grad():
            valid_stat = run_epoch(model, core, rounded_readout,
                                   param_weights, geom_weight, valid_ldr,
                                   desc           = f'Valid Epoch {epoch} / {num_epochs}',
                                   device         = device,
                                   is_distributed = is_distributed,
                                   is_main        = is_main)

        scheduler.step()

        if is_main:
            checkpointer.save(epoch)

            for log, stat in zip([train_log, valid_log], [train_stat, valid_stat]):
                stat.update({'lr': current_lr, 'epoch': epoch})
                frame = pd.DataFrame(data=stat, index=[1])
                frame.to_csv(log, index=False, float_format='%.6g',
                             mode   = 'a' if log.exists() else 'w',
                             header = not log.exists())

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    train()

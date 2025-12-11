import argparse
import time
from pathlib import Path
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True # <- may be not necessary

# cpu -> gpu transfer time or not

from rtal.datasets.dataset import ROMDataset
from mlp import MLP


parser = argparse.ArgumentParser('throughput test')

parser.add_argument('--config',
                    type    = str,
                    default = 'config.yaml',
                    help    = 'path to config file | config.yaml')
parser.add_argument('--device',
                    type    = str,
                    default = 'cuda',
                    choices = ('cuda', 'cpu'),
                    help    = 'device to train the model on | default = cuda')
parser.add_argument('--gpu-id',
                    type    = int,
                    default = 0,
                    help    = 'GPU to train the model on | default = 0')
parser.add_argument('--batch-size',
                    type    = int,
                    default = 1,
                    help    = 'batch_size | default = 1')
parser.add_argument('--n-runs',
                    type    = int,
                    default = 1000,
                    help    = 'number of runs | default = 1000')
parser.add_argument('--output',
                    type    = str,
                    default = 'throughput.csv',
                    help    = 'output file')
parser.add_argument('--matmul-precision',
                    type    = str,
                    default = 'high',
                    choices = ('highest', 'high', 'medium'),
                    help    = ('The parameter to '
                               'torch.set_float32_matmul_precision '
                               '| default = high'))
parser.add_argument('--on-gpu',
                    action  = 'store_true',
                    help    = ('Whether to send the data to GPU beforehand. '
                               'Only useful when device==cuda'))

args = parser.parse_args()

# == device and core ======================================================
device = args.device
if device == 'cuda':
    torch.cuda.set_device(args.gpu_id)

torch.set_float32_matmul_precision(args.matmul_precision)

# == model ================================================================
with open(args.config, 'r', encoding='UTF-8') as handle:
    config = yaml.safe_load(handle)

model = MLP(**config['model'])
model = model.to(device)
model.eval()
model = torch.compile(model)

# == dataloader ===========================================================
data_root = '/data/yhuang2/rtal/rom_det-3_part-200_cont-and-rounded/'
dataset  = ROMDataset(data_root, split='train', num_particles=50)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

# == readout ==============================================================
event = next(iter(dataloader))
readout = event['readout_curr_cont']
readout = torch.transpose(readout, 1, 2).flatten(-2, -1)

print(readout.shape)

if device == 'cuda':
    if args.on_gpu:
        readout = readout.to(device)

# == warm-up ==============================================================
with torch.no_grad():
    for _ in range(10):
        if device == 'cuda' and (not args.on_gpu):
            _ = model(readout.to(device))
        else:
            _ = model(readout)

if device == "cuda":
    torch.cuda.synchronize()

# == timing ===============================================================
if device == 'cuda':
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        for event in tqdm(range(args.n_runs)):
            if not args.on_gpu:
                _ = model(readout.to(device))
            else:
                _ = model(readout)

    end_event.record()
    torch.cuda.synchronize()

    ms = start_event.elapsed_time(end_event)
else:
    # CPU timing
    start_time = time.perf_counter()

    with torch.no_grad():
        for event in tqdm(range(args.n_runs)):
            _ = model(readout)

    end_time = time.perf_counter()
    ms = (end_time - start_time) * 1000

# millisecond per event
ms_per_event = ms / (args.n_runs * args.batch_size)
print(f"Average inference time: {ms_per_event:.8f} ms per sample")

# == output ===============================================================
output = Path(args.output)
if not output.exists():
    with open(output, 'w', encoding='UTF-8') as handle:
        handle.write('device,'
                     'gpu_id,'
                     'matmul_precision,'
                     'on_gpu,'
                     'batch_size,'
                     'n_runs,'
                     'ms_per_event\n')

with open(output, 'a', encoding='UTF-8') as handle:
    handle.write(f'{args.device},'
                 f'{args.gpu_id},'
                 f'{args.matmul_precision},'
                 f'{args.on_gpu},'
                 f'{args.batch_size},'
                 f'{args.n_runs},'
                 f'{ms_per_event:.8f}\n')

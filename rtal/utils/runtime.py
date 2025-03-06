from time import time
from tqdm import tqdm

import torch
from torch.jit import trace

def runtime(model,
            input_shape,
            batch_size,
            num_inference_batches,
            num_warmup_batches = 10,
            script = False,
            device = 'cuda'):

    """
    Input
    ========
    1. model:
    1. input_shape:
    1. batch_size:
    1. num_inference_batches:
    1. num_warmup_batches:
    1. script:
    """

    model = model.to(device)
    data = torch.randn((batch_size, ) + input_shape).to(device)

    if script:
        model = trace(model, data)


    with torch.no_grad():
        for _ in tqdm(range(num_warmup_batches), desc='warmup'):
            model(data)

        time0 = time()
        for _ in tqdm(range(num_inference_batches), desc='inerence'):
            model(data)

    time_elapsed = time() - time0
    sample_per_second = batch_size * num_inference_batches / time_elapsed

    return sample_per_second

import sys
import yaml

import torch

from mlp import MLP

config_fname = sys.argv[1]
print(config_fname)

with open(config_fname, 'r', encoding='UTF-8') as handle:
    config = yaml.safe_load(handle)

model = MLP(**config['model'])

model.eval()

# Create dummy input with the correct shape
in_features = config['model']['in_features']
num_particles = config['data']['num_particles']
dummy_input = torch.randn(1, num_particles, in_features)

# Step 4: Export to ONNX
torch.onnx.export(
    model,                      # model being run
    dummy_input,                # model input (or a tuple for multiple inputs)
    "mlp.onnx",                 # where to save the model (filename)
    export_params=True,         # store the trained weights inside the model
    opset_version=11,           # the ONNX version to export to (11 is widely supported)
    do_constant_folding=True,   # optimize constants
    input_names=['input'],      # input name (can be arbitrary)
    output_names=['output'],    # output name
    dynamic_axes={              # support dynamic batch size
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
)

print("Model has been exported to ONNX format.")

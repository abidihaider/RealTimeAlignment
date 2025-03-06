"""
Load ROM Dataset
"""

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

class ROMDataset(Dataset):
    """
    Dataset API for loading the toy-detector dataset
    """
    def __init__(self, data_root, num_particles):
        super().__init__()
        self.fnames = list(Path(data_root).glob('*npz'))
        self.fnames = sorted(self.fnames, key=lambda fname: int(fname.stem.split('_')[-1]))
        self.num_particles = num_particles

    def __len__(self,):
        return len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]

        with np.load(fname) as handle:
            detector_start     = handle['detector_start']
            detector_curr      = handle['detector_curr']
            readout_start      = handle['readout_start']
            readout_curr       = handle['readout_curr']
            particle_vertex    = handle['particle_vertex']
            particle_direction = handle['particle_direction']

        indices = np.random.permutation(particle_vertex.shape[0])[:self.num_particles]

        return {'detector_start'     : detector_start.astype(np.float32),
                'detector_curr'      : detector_curr.astype(np.float32),
                'readout_start'      : readout_start[:, indices].astype(np.float32),
                'readout_curr'       : readout_curr[:, indices].astype(np.float32),
                'particle_vertex'    : particle_vertex[indices].astype(np.float32),
                'particle_direction' : particle_direction[indices].astype(np.float32)}


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    dataset = ROMDataset('../../notebooks/demo_dataset', num_particles=50)
    dataloader = DataLoader(dataset, batch_size=4)

    for sample in dataloader:
        for key, val in sample.items():
            print(f"{key + ':':20s}", val.shape)
        break

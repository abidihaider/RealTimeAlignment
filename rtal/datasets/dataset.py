"""
Load ROM Dataset
"""

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


class ROMDataset(Dataset):
    """
    Dataset API for loading the toy-detector dataset
    """
    def __init__(self,
                 data_root,
                 num_particles,
                 split = None,
                 mode  = 'raw'):

        super().__init__()

        available_modes = ('raw', 'tracked', 'point_cloud')
        if mode == 'raw':
            self.readout_processor = self.__get_raw_readout
        elif mode == 'tracked':
            self.readout_processor = self.__get_tracked_readout
        elif mode == 'point_cloud':
            self.readout_processor = self.__get_point_cloud_readout
        else:
            print(f'Unknown mode {mode}! Choose from {available_modes}.')

        data_root = Path(data_root)

        if split is not None:
            data_root = data_root/split

        self.fnames = list(data_root.glob('*npz'))
        self.fnames = sorted(self.fnames, key=lambda fname: int(fname.stem.split('_')[-1]))
        self.num_particles = num_particles

    @staticmethod
    def __get_raw_readout(readout):
        return readout

    @staticmethod
    def __get_tracked_readout(readout):
        readout = np.transpose(readout, (1, 0, 2))
        num_particles = len(readout)
        return readout.reshape(num_particles, -1)

    @staticmethod
    def __get_point_cloud_readout(readout):
        return np.vstack([np.hstack([(det_id + 1) * np.ones((len(rd), 1)), rd])
                          for det_id, rd in enumerate(readout)])

    def __len__(self,):
        return len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]

        results = {}
        with np.load(fname) as handle:
            # detector parameters
            results['detector_start'] = handle['detector_start']
            results['detector_curr']  = handle['detector_curr']

            # particles
            particle_vertex    = handle['particle_vertex']
            particle_direction = handle['particle_direction']

            # select particles
            indices = np.random.permutation(particle_vertex.shape[0])[:self.num_particles]

            results['particle_vertex'] = particle_vertex[indices]
            results['particle_direction'] = particle_direction[indices]

            # readouts
            for key in handle.keys():
                if 'readout' in key:
                    val = handle[key]
                    results[key] = self.readout_processor(val[:, indices])

        return {key: val.astype(np.float32) for key, val in results.items()}


def to_cloud(readout):
    """
    Input:
        - readout shape (batch_size, num_particles, num_dets, 2)
    output
        - shape (batch_size, num_dets * num_particles, 3)
    """

    device = readout.device

    batch_size, num_particles, num_dets = readout.shape[:-1]
    # det_ids = [1, 2, ..., num_dets]
    det_ids = torch.arange(1, num_dets + 1, device=device)
    # det_ids: (num_dets,) -> (num_dets * num_particles,)
    # -> (num_dets * num_particles, 1)
    # -> (batch_size, num_dets * num_particles, 1)
    det_ids = det_ids.repeat_interleave(num_particles)\
                     .unsqueeze(-1)\
                     .expand(batch_size, num_dets * num_particles, 1)
    return torch.cat([det_ids, readout.flatten(start_dim=1, end_dim=2)], dim=-1)


def to_track(readout):
    """
    Input:
        - readout shape (batch_size, num_dets, num_particles, 2)
    output
        - shape (batch_size, num_particles, num_dets * 2)
    """
    return readout.transpose(1, 2).flatten(start_dim=-2, end_dim=-1)


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    def test(root):

        for mode in ('raw', 'tracked', 'point_cloud'):
            np.random.seed(0)
            print(f'== {mode} ==========')
            dataset = ROMDataset(root,
                                 mode=mode,
                                 num_particles=1)
            dataloader = DataLoader(dataset, batch_size=2)

            for sample in dataloader:
                for key, val in sample.items():
                    print(f"{key + ':':20s}", val.shape)

                for key in sample:
                    if 'readout' in key:
                        readout = sample[key]
                        print(f'{key}\n{readout}')

                        if mode == 'raw':
                            print(f'cloud\n{to_cloud(readout)}')
                            print(f'track\n{to_track(readout)}')
                break
            print()

    print()
    test('/data/yhuang2/rtal/rom_det-3_part-200_rounded/train')

    print()
    test('/data/yhuang2/rtal/rom_det-3_part-200_cont-and-rounded/train')

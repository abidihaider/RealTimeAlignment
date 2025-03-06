"""
Generate dataset
"""
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

import numpy as np

from detector import Detector
from particle import RandomParticle


def generate_one(config, output_folder, fname):
    """
    Generate one dataset
    """

    # == set random seed ======================================================
    np.random.seed(config['dataset']['random_seed'])

    # create detectors
    detectors = []
    for dt_config in config['detectors']:
        detector = Detector(**dt_config)
        detectors.append(detector)

    # == misalign the detectors ===============================================
    max_num_misalignments = config['misalignments']['max_num_misalignments']
    all_misalignments = list(config['misalignments'].keys())
    all_misalignments.remove('max_num_misalignments')

    for detector in detectors:
        num_misalignments = np.random.randint(1, max_num_misalignments + 1)
        misalignments = np.random.choice(all_misalignments, num_misalignments)

        for misal in misalignments:
            kwargs = config['misalignments'][misal]
            detector.misalign(misal, kwargs, verbose=False)

    # == generate random particles ============================================
    particle_generator = RandomParticle(**config['particles'])
    num_particles = config['dataset']['num_particles']
    particles = particle_generator(num_particles)

    # == calculate readout ====================================================
    mask = np.ones(num_particles, dtype=bool)
    readout_start = []
    readout_curr  = []
    rounded = config['dataset']['rounded']
    for detector in detectors:
        rd_start, mask_start = detector.get_readout(particles,
                                                    state   = 'start',
                                                    rounded = rounded)
        rd_curr,  mask_curr  = detector.get_readout(particles,
                                                    state   = 'curr',
                                                    rounded = rounded)
        readout_start.append(rd_start)
        readout_curr.append(rd_curr)
        mask &= mask_start & mask_curr

    readout_start = np.stack([item[mask] for item in readout_start])
    readout_curr = np.stack([item[mask] for item in readout_curr])

    # == save =================================================================
    dt_params = [detector.get_parameters() for detector in detectors]
    np.savez_compressed(
        output_folder/fname,
        detector_start     = np.stack([param['start'] for param in dt_params]),
        detector_curr      = np.stack([param['curr'] for param in dt_params]),
        readout_start      = readout_start,
        readout_curr       = readout_curr,
        particle_vertex    = particles.vertex[mask],
        particle_direction = particles.direction[mask]
    )


def generate_dataset(num_samples, config_fname, output_folder):
    """
    Generate a dataset
    """

    # load config
    with open(config_fname, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    # create dataset folder
    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    else:
        raise ValueError(f'{output_folder} is not empty!')

    # create samples
    for sample_idx in tqdm(range(num_samples)):
        fname = f'sample_{sample_idx}'
        generate_one(config, output_folder, fname)


def main():
    """
    Take dataset arguments and genearte a dataset.
    """

    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument('--num-samples',
                        type    = int,
                        default = 1,
                        help    = 'number of samples')
    parser.add_argument('--output-folder',
                        type    = str,
                        default = './',
                        help    = 'output folder')
    parser.add_argument('--config',
                        type    = str,
                        default = 'config.yaml',
                        help    = 'path to the config file.')

    args = parser.parse_args()

    generate_dataset(num_samples   = args.num_samples,
                     config_fname  = args.config,
                     output_folder = args.output_folder)


if __name__ == '__main__':
    main()

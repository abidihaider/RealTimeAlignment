from itertools import product
from pathlib import Path
import yaml


def assemble(num_features,
             num_subset_solvers,
             subset_size,
             subset_solver_depth,
             rezero):

    model_config = {
        'embedding_features': [num_features, num_features],
        'subset_config': [[subset_size] + [num_features] * subset_solver_depth] * num_subset_solvers,
        'rezero': rezero
    }
    return model_config


def make_name(mydict):
    return '|'.join(f'{key}-{val}' for key, val in mydict.items())


def model_config(config_grid):
    keys = config_grid.keys()
    values = config_grid.values()
    for param in product(*values):
        param_dict = {key: val for key, val in zip(keys, param)}
        fname = make_name(param_dict)
        yield assemble(**param_dict), fname


def main():

    job_root = Path('job_root')
    job_root.mkdir(parents=True, exist_ok=True)

    with open('config_template.yaml', 'r') as handle:
        config_template = yaml.safe_load(handle)

    with open('grid.yaml', 'r') as handle:
        config_grid = yaml.safe_load(handle)

    updates = model_config(config_grid)

    for i, (update, job_name) in enumerate(updates):
        config = config_template.copy()
        config['model'].update(update)

        job_path = job_root/job_name
        job_path.mkdir(parents=True, exist_ok=True)

        with open(job_path/'config.yaml', 'w') as handle:
            yaml.dump(config, handle)

if __name__ == '__main__':
    main()

from itertools import product
from pathlib import Path
import yaml


def assemble(num_features,
             num_subset_solvers,
             subset_size,
             subset_solver_depth,
             rounded):

    model_config = {
        'embedding_features': [num_features, num_features],
        'subset_config': [[subset_size] + [num_features] * subset_solver_depth] * num_subset_solvers,
    }
    data_config = {'rounded': rounded}
    return {'model': model_config,
            'data': data_config}


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

    jobs_per_folder = 24

    with open('config_template.yaml', 'r') as handle:
        config_template = yaml.safe_load(handle)

    with open('grid.yaml', 'r') as handle:
        config_grid = yaml.safe_load(handle)

    updates = model_config(config_grid)

    for i, (update, job_name) in enumerate(updates):
        if i % jobs_per_folder == 0:
            job_root = Path(f'../frontier/job_roots/job_root_{i // jobs_per_folder}')
            job_root.mkdir(parents=True, exist_ok=True)

        config = config_template.copy()
        for key, val in update.items():
            config[key].update(val)

        job_path = job_root/f'mlp|{job_name}'
        job_path.mkdir(parents=True, exist_ok=True)

        with open(job_path/'config.yaml', 'w') as handle:
            yaml.dump(config, handle)

if __name__ == '__main__':
    main()

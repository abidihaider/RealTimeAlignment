"""
Checkpointing facility
"""
from pathlib import Path
import torch


class Checkpointer:
    """
    Checkpointing the model and optionally optimizer and scheduler
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 model, *,
                 optimizer       = None,
                 scheduler       = None,
                 checkpoint_path ='./',
                 save_frequency  = None,
                 prefix          = 'ckpt'):
        """
        Description:
            Save and load checkpoints.
            The save function will always save the current model. The
            epoch of the current model will be saved to
            [checkpoint_path]/last_saved_epoch.
            If [save_frequency] is given, also save every [save_frequency]
            epochs.
        Input:
            - checkpoint_path: location of checkpoints
            - save_frequency: the checkpoints are saved every
              [save_frequency] epochs.
            - submodel_names: a list-like object in which all
              submodels that need to be saved individually are listed
            - model_prefix: the prefix used for save the full model
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.prefix = prefix

        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        # We need a file to save the last saved epoch
        # this helps us figure out the epoch to resume
        self.last_saved_fname = self.checkpoint_path/'last_saved_epoch'

        self.save_frequency = save_frequency

    def __save(self, suffix):
        """
        Save (sub)model checkpoints.
        """
        model_path = self.checkpoint_path/f'{self.prefix}_{suffix}.path'
        # assert not model_path.exists(), \
        #     f'{model_path} already exists, refuse to overwrite'

        # checkpoint dictionary
        checkpoint = {'model': self.model.state_dict()}

        if self.optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        torch.save(checkpoint, model_path)

    def save(self, epoch):
        """
        Save the latest and every save_frequency paths.
        """

        self.__save(suffix='last')
        with open(self.last_saved_fname, 'w', encoding='UTF-8') as handle:
            handle.write(f'{epoch}')

        if self.save_frequency is not None:
            if epoch % self.save_frequency == 0:
                self.__save(suffix=epoch)

    def load(self, epoch='last', prefix=None, device='cuda'):
        """
        Epoch should be a positive integer or the string 'last'
        """
        assert (isinstance(epoch, int) and (epoch > 0)) or (epoch == 'last'), \
            "epoch should either be a positive integer or the string 'last'"

        # get last saved epoch number
        last_saved_epoch = 0
        if self.last_saved_fname.exists():
            with open(self.last_saved_fname, 'r', encoding='UTF-8') as handle:
                for line in handle:
                    last_saved_epoch = int(line.strip())
                    break

        if last_saved_epoch == 0:
            print('Train from scratch')
            return 0


        prefix = self.prefix if prefix is None else prefix

        # Load a checkpoint
        model_path = self.checkpoint_path/f'{prefix}_{epoch}.path'

        assert model_path.exists(), \
            f'Requested checkpoint {str(model_path)} does not exist!'

        print(f'Load model {str(model_path)}')

        checkpoint = torch.load(model_path, map_location=device)

        self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        # make sure all tensors are on the same device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        return last_saved_epoch if epoch == 'last' else epoch

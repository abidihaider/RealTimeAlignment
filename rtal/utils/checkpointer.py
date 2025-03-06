"""
Checkpointing facility
"""
from pathlib import Path
import torch


class Checkpointer:
    def __init__(self,
                 model,
                 checkpoint_path,
                 save_frequency=None,
                 model_prefix='model',
                 submodel_names=None):
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

        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        # We need a file to save the last saved epoch
        # this helps us figure out the epoch to resume
        self.last_saved_fname = self.checkpoint_path/'last_saved_epoch'

        self.save_frequency = save_frequency

        self.submodel_names = [] if submodel_names is None else submodel_names

        if model_prefix is None:
            self.model_prefix = 'model'
        else:
            self.model_prefix = model_prefix

        if self.submodel_names:
            assert self.model_prefix not in self.submodel_names, \
                ("Full model prefix ({self.model_prefx}) should "
                 "not be in submodel names ({self.submodel_names})!")

    def __save(self, suffix):
        """
        Save (sub)model checkpoints.
        """
        model_pth = self.checkpoint_path/f'{self.model_prefix}_{suffix}.pth'
        torch.save(self.model.state_dict(), model_pth)

        for submodel_name in self.submodel_names:
            submodel = getattr(self.model, submodel_name)
            submodel_pth = self.checkpoint_path/f'{submodel_name}_{suffix}.pth'
            torch.save(submodel.state_dict(), submodel_pth)

    def save(self, epoch):
        """
        Save the latest and every save_frequency pths.
        """

        self.__save(suffix='last')
        with open(self.last_saved_fname, 'w', encoding='UTF-8') as handle:
            handle.write(f'{epoch}')

        if self.save_frequency is not None:
            if epoch % self.save_frequency == 0:
                self.__save(suffix=epoch)

    def load(self, epoch='last', prefix=None):
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

        prefix = self.model_prefix if prefix is None else prefix

        # Load a checkpoint
        model_pth = self.checkpoint_path/f'{prefix}_{epoch}.pth'

        assert model_pth.exists(), \
            f'Requested checkpoint {str(model_pth)} does not exist!'

        print(f'Load model {str(model_pth)}')

        self.model.load_state_dict(torch.load(model_pth))

        return last_saved_epoch if epoch == 'last' else epoch

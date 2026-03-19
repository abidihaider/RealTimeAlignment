"""
Simple QAT-aware checkpointer (Option A: save training + converted inference checkpoints)

Saves:
  - {prefix}_last.pth               (training / prepared model)
  - {prefix}_{epoch}.pth            (every save_frequency epochs, if set)
  - {prefix}_last_converted.pth     (converted inference model)
  - {prefix}_{epoch}_converted.pth  (every save_frequency epochs, if set)

Assumptions:
  - self.model is the *prepared* model (after qat_quantizer.prepare()) for training.
  - qat_quantizer.convert(model_copy) produces a converted inference model.
"""

from pathlib import Path
import os
import time
import copy
import torch


class CheckpointerQAT:
    def __init__(self,
                 model, *,
                 optimizer       = None,
                 scheduler       = None,
                 qat_quantizer   = None,
                 checkpoint_path = './',
                 save_frequency  = None,
                 prefix          = 'ckpt'):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.qat_quantizer = qat_quantizer

        self.prefix = prefix

        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.last_saved_fname = self.checkpoint_path / 'last_saved_epoch'
        self.save_frequency = save_frequency

    # ---------- small helpers ----------
    def _atomic_torch_save(self, obj, path: Path):
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _atomic_write_text(self, path: Path, text: str):
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, path)

    def _ckpt_path(self, suffix: str) -> Path:
        return self.checkpoint_path / f'{self.prefix}_{suffix}.pth'

    def _get_last_saved_epoch(self) -> int:
        if not self.last_saved_fname.exists():
            return 0
        try:
            with open(self.last_saved_fname, 'r', encoding='utf-8') as f:
                s = f.readline().strip()
            return int(s) if s else 0
        except Exception:
            return 0

    def _move_optimizer_state_to(self, device: str):
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ---------- save ----------
    def _save_training(self, epoch: int, suffix: str):
        path = self._ckpt_path(suffix)
        ckpt = {
            'model': self.model.state_dict(),
            'meta': {
                'epoch': epoch,
                'timestamp': time.time(),
                'torch_version': torch.__version__,
                'kind': 'qat_training_prepared',
            }
        }
        if self.optimizer is not None:
            ckpt['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            try:
                ckpt['scheduler'] = self.scheduler.state_dict()
            except Exception:
                ckpt['scheduler'] = None

        # quantizer state/config is optional; store if it exists
        if self.qat_quantizer is not None and hasattr(self.qat_quantizer, 'state_dict'):
            try:
                ckpt['qat_quantizer'] = self.qat_quantizer.state_dict()
            except Exception:
                ckpt['qat_quantizer'] = None

        self._atomic_torch_save(ckpt, path)
        return path

    def _save_converted(self, epoch: int, suffix: str, convert_on: str = 'cpu'):
        if self.qat_quantizer is None or not hasattr(self.qat_quantizer, 'convert'):
            raise RuntimeError("Need qat_quantizer with .convert(model) to save converted checkpoints.")

        # Convert a copy so we don't touch the training model.
        model_copy = copy.deepcopy(self.model).to('cpu')
        converted = self.qat_quantizer.convert(model_copy)
        converted = converted.to(convert_on)

        path = self._ckpt_path(suffix)
        ckpt = {
            'model': converted.state_dict(),
            'meta': {
                'epoch': epoch,
                'timestamp': time.time(),
                'torch_version': torch.__version__,
                'kind': 'inference_converted',
            }
        }
        self._atomic_torch_save(ckpt, path)
        return path

    def save(self, epoch: int, save_converted: bool = True, convert_on: str = 'cpu'):
        """
        Save latest checkpoint and, if save_frequency is set, also save periodic snapshots.
        """
        # always save last
        self._save_training(epoch, suffix='last')
        self._atomic_write_text(self.last_saved_fname, str(epoch))

        if save_converted:
            self._save_converted(epoch, suffix='last_converted', convert_on=convert_on)

        # periodic
        if self.save_frequency is not None and self.save_frequency > 0 and epoch % self.save_frequency == 0:
            self._save_training(epoch, suffix=str(epoch))
            if save_converted:
                self._save_converted(epoch, suffix=f'{epoch}_converted', convert_on=convert_on)

    # ---------- load (training / prepared) ----------
    def load(self, epoch='last', device='cuda'):
        """
        Load training checkpoint into self.model (+ optimizer/scheduler if present).
        Returns epoch number to resume from (0 means start from scratch).
        """
        assert (isinstance(epoch, int) and epoch > 0) or (epoch == 'last'), \
            "epoch should be a positive int or 'last'"

        last_saved_epoch = self._get_last_saved_epoch()
        if epoch == 'last' and last_saved_epoch == 0:
            print('Train from scratch')
            return 0

        path = self._ckpt_path(str(epoch))
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')

        ckpt = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt['model'], strict=True)
        self.model.to(device)

        if self.optimizer is not None and 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self._move_optimizer_state_to(device)

        if self.scheduler is not None and 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except Exception:
                print('Warning: scheduler state could not be restored cleanly.')

        # quantizer restore if supported (not always necessary)
        if self.qat_quantizer is not None and 'qat_quantizer' in ckpt and ckpt['qat_quantizer'] is not None:
            if hasattr(self.qat_quantizer, 'load_state_dict'):
                try:
                    self.qat_quantizer.load_state_dict(ckpt['qat_quantizer'])
                except Exception:
                    pass

        return last_saved_epoch if epoch == 'last' else epoch

    def load_converted_into(self, model, epoch='last', device='cuda', strict=True):
        """
        Load a converted inference checkpoint into `model` (a compatible converted model instance).
        Returns the epoch stored in the checkpoint metadata (or 0).
        """
        assert (isinstance(epoch, int) and epoch > 0) or (epoch == 'last'), \
            "epoch should be a positive int or 'last'"

        suffix = 'last_converted' if epoch == 'last' else f'{epoch}_converted'
        path = self._ckpt_path(suffix)
        if not path.exists():
            raise FileNotFoundError(f'Converted checkpoint not found: {path}')

        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=strict)
        model.to(device)
        model.eval()

        meta = ckpt.get('meta', {})
        return int(meta.get('epoch', 0) or 0)

    def load_converted(self, base_model, epoch='last', device='cuda', strict=True):
        """
        Convenience: given a freshly initialized *float* base_model (from config),
        this will:
          1) prepare(base_model)
          2) convert(prepared_model)  -> gives correct converted structure
          3) load converted checkpoint weights into it
          4) move to device + eval
        Returns: converted inference model
        """
        if self.qat_quantizer is None:
            raise RuntimeError("qat_quantizer is required for load_converted().")

        if not hasattr(self.qat_quantizer, 'prepare') or not hasattr(self.qat_quantizer, 'convert'):
            raise RuntimeError("qat_quantizer must have .prepare() and .convert().")

        # Build the converted-structure model from scratch to match the saved converted state_dict
        m = self.qat_quantizer.prepare(base_model)
        m = self.qat_quantizer.convert(m)

        # Load weights from the converted checkpoint
        self.load_converted_into(m, epoch=epoch, device=device, strict=strict)
        return m

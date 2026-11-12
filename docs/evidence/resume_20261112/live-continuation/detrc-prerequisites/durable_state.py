"""Epoch-boundary native MMCV checkpoint plus DeTRC Python/RNG recovery.

This hook does not alter optimization/selection. A crash inside an epoch replays
that incomplete epoch from the last committed boundary. GPU RNG is captured only
inside an authorized real run, never by CPU preparation.
"""
import os
import random
from pathlib import Path
import numpy as np
import torch
from mmcv.runner import HOOKS, Hook


def bare(model):
    return model.module if hasattr(model, 'module') else model


def runtime_state(model):
    m = bare(model)
    if m.use_position_embedding == 'learnable':
        if torch.count_nonzero(m.pos_embed.weight).item() != 0:
            raise RuntimeError('nonzero default TSN position table')
    return dict(cls_warmup_step=m.cls_warmup_step,
                coef_ce_now=m.coef_ce_now, coef_count_now=m.coef_count_now,
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_runtime(model, state):
    m = bare(model)
    if m.use_position_embedding == 'learnable' and torch.count_nonzero(m.pos_embed.weight).item() != 0:
        raise RuntimeError('nonzero default TSN position table on restore')
    for key in ('cls_warmup_step', 'coef_ce_now', 'coef_count_now'):
        setattr(m, key, state[key])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


@HOOKS.register_module()
class DeTRCDurableStateHook(Hook):
    def before_run(self, runner):
        # MMCV resume restores model, optimizer, epoch/iter and hook_msgs in meta.
        if runner.epoch > 0:
            state = runner.meta.get('detrc_runtime') if runner.meta else None
            if state is None:
                raise RuntimeError('resume requires a durable DeTRC epoch checkpoint')
            restore_runtime(runner.model, state)
        runtime_state(runner.model)

    def after_train_epoch(self, runner):
        if runner.meta is None:
            runner.meta = {}
        runner.meta['detrc_runtime'] = runtime_state(runner.model)
        # Step LR is recomputed from persisted epoch/iter and optimizer initial_lr;
        # MMCV hook state (including best validation OBO) is in runner.meta.
        tmp = 'durable_epoch_pending.pth'
        runner.save_checkpoint(runner.work_dir, filename_tmpl=tmp,
                               save_optimizer=True, create_symlink=False)
        src = Path(runner.work_dir) / tmp
        with src.open('rb') as f:
            os.fsync(f.fileno())
        dst = Path(runner.work_dir) / 'durable_latest.pth'
        os.replace(src, dst)
        directory = os.open(runner.work_dir, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def atomic_native_best(self, runner, key_score):
    """Identical native comparison; commit bytes before the selected path."""
    if not self.by_epoch:
        raise RuntimeError('fixed DeTRC route uses epoch selection')
    old_msgs = dict(runner.meta['hook_msgs'])
    old_score = old_msgs.get('best_score', self.init_value_map[self.rule])
    if not self.compare_func(key_score, old_score):
        return
    name = f'best_{self.key_indicator}_epoch_{runner.epoch + 1}.pth'
    out = Path(self.out_dir)
    selected = out / name
    runner.meta['hook_msgs']['best_score'] = key_score
    runner.meta['hook_msgs']['best_ckpt'] = str(selected)
    runner.meta['detrc_runtime'] = runtime_state(runner.model)
    tmp = name + '.pending'
    try:
        runner.save_checkpoint(str(out), filename_tmpl=tmp,
                               save_optimizer=True, create_symlink=False)
        with (out / tmp).open('rb') as f:
            os.fsync(f.fileno())
        os.replace(out / tmp, selected)
        directory = os.open(out, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        runner.meta['hook_msgs'] = old_msgs
        raise
    self.best_ckpt_path = str(selected)
    runner.logger.info('Atomically selected native best %s=%s at epoch %s',
                       self.key_indicator, key_score, runner.epoch + 1)


# Only the candidate process imports this module; shared package files untouched.
from mmaction.core import EvalHook
EvalHook._save_ckpt = atomic_native_best

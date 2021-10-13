# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class RoiEpochSetHook(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def before_epoch(self, runner):
        if hasattr(runner.model.module.roi_head.bbox_head, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.model.module.roi_head.bbox_head.set_epoch(runner.epoch)
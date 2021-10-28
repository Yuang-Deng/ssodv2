# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm

# def save(images, annotations):
#     ann = {}
#     ann['type'] = 'instances'
#     ann['images'] = images
#     ann['annotations'] = annotations

#     categories = []
#     for k, v in label_ids.items():
#         categories.append({"name": k, "id": v})
#     ann['categories'] = categories
#     json.dump(ann, open('images/instances_test.json', 'w'))


# def test_dataset(im_dir):
#     # im_list = glob(os.path.join(im_dir, '*.jpg'))
#     im_list = os.listdir(im_dir)

#     im_list.sort(key=lambda x: int(x.split('.')[0]))
#     idx = 0
#     image_id = -1
#     images = []
#     annotations = []
#     for im_path in tqdm(im_list):
#         image_id += 1
#         im = Image.open(os.path.join(im_dir,im_path))
#         w, h = im.size
#         image = {'file_name': os.path.basename(im_path), 'width': w, 'height': h, 'id': image_id}
#         images.append(image)
#         labels = [[10, 10, 20, 20]]
#         for label in labels:
#             bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
#             seg = []
#             ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
#                    'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
#             idx += 1
#             annotations.append(ann)
#     save(images, annotations)

class EvalHook(BaseEvalHook):

    def before_train_epoch(self, runner):
        self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False, return_meta=True)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)

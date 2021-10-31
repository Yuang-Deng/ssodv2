# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
import numpy as np

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer


@HEADS.register_module()
class GMMBBoxHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 eta=12,
                 cls_lambda=1,
                 lam_box_loss=1,
                 cls_warm_epoch=2,
                 box_warm_epoch=2,
                 cls_pos_lambda=1,
                 cls_neg_lambda=1,
                 cls_unc_type='al',
                 box_unc_type='al',
                 lambda_unc_box=1,
                 lambda_unc_cls=1,
                 init_cfg=None):
        super(GMMBBoxHead, self).__init__(init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False
        self.eta = eta
        self.cls_lambda = cls_lambda
        self.lam_box_loss = lam_box_loss
        self.cls_warm_epoch = cls_warm_epoch
        self.box_warm_epoch = box_warm_epoch
        self.cls_pos_lambda = cls_pos_lambda
        self.cls_neg_lambda = cls_neg_lambda
        self.cls_unc_type = cls_unc_type
        self.box_unc_type = box_unc_type
        self.lambda_unc_box = lambda_unc_box
        self.lambda_unc_cls = lambda_unc_cls    

        self.cls_gt_al = [0.011676047928631306, 0.013194473460316658, 0.004626300651580095, 0.011304869316518307, 0.008983615785837173, 0.012291173450648785, 0.015498469583690166, 0.015249814838171005, 0.007545147091150284, 0.010576263070106506, 0.01383853517472744 , 0.010896758176386356, 0.007504409644752741, 0.008247287012636662, 0.003804341424256563, 0.006531019229441881, 0.006469824817031622, 0.017160730436444283, 0.01712864451110363 , 0.015580035746097565]
        self.cls_gt_ep = [0.0014669252559542656, 0.0010269444901496172, 0.001524801948107779, 0.0021521486341953278, 0.0010409681126475334, 0.0021159478928893805, 0.015673134475946426, 0.002468044636771083, 0.0035432521253824234, 0.001807992230169475, 0.0003179576597176492, 0.004923964384943247, 0.003719983622431755, 0.0011385264806449413, 0.00708423275500536, 0.0009856268297880888, 0.002989743836224079, 0.000680554483551532, 0.0006094087148085237, 0.003230171510949731]
        self.box_gt_al = [0.4997519850730896 , 0.4818139374256134 , 0.41686928272247314, 0.6221621632575989 , 0.8729903697967529 , 0.38314908742904663, 0.25480929017066956, 0.45177340507507324, 0.6295313835144043 , 0.44437214732170105, 0.8023167848587036 , 0.3742786645889282 , 0.4825665056705475 , 0.45643332600593567, 0.5225471258163452 , 0.7808040380477905 , 0.5682315826416016 , 0.6573629975318909 , 0.42482975125312805, 0.35368612408638   ]
        self.box_gt_ep = [0.10289988666772842, 0.139907568693161  , 0.13576948642730713, 0.15829303860664368, 0.15875084698200226, 0.16106688976287842, 0.11345464736223221, 0.09817182272672653, 0.25670957565307617, 0.1447836458683014 , 0.3406357169151306 , 0.10156666487455368, 0.15055552124977112, 0.11262432485818863, 0.22155672311782837, 0.23454366624355316, 0.17912887036800385, 0.1696951687335968 , 0.08456728607416153, 0.12094420194625854]



        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_unc = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=in_channels,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=in_channels,
                out_features=out_dim_reg)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def _gaussian_dist_pdf(self, val, mean, var):
        eps = 0.3
        return torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * (var + eps**2))
    
    def gen_one_hot_label(self, num_classes, labels, device):
        one_hot_label = torch.zeros(labels.size(0), num_classes + 1).to(device)
        # index = torch.arange(0, labels.size(0), 1).to(device)
        # index = torch.cat([index[:, None], labels[:, None]], dim=-1)
        # one_hot_label[index] = 1
        one_hot_label.scatter_(1, labels[:, None], 1)
        return one_hot_label

    def get_unc(self, bbox_pred):
        cls_num = self.num_classes
        # xishu softmax    fangcha relu
        box_rep_len = cls_num * 4
        # box_rep_len = (cls_score.size(1) - 1) * 4
        gmm_k =  bbox_pred.size(1) // (box_rep_len * 3)
        bbox_pred = bbox_pred.view(bbox_pred.size(0), 3, -1)
        mu_box = bbox_pred[:, 0, :].view(bbox_pred.size(0), gmm_k, -1)
        sigma_box = bbox_pred[:, 1, :].view(bbox_pred.size(0), gmm_k, -1)
        pi_box = bbox_pred[:, 2, :].view(bbox_pred.size(0), gmm_k, -1)
        mu_al_box = (pi_box * sigma_box).sum(dim=1)
        mu_ep_box = (pi_box * torch.pow(mu_box - (pi_box * mu_box).sum(dim=1)[:, None, :].expand(bbox_pred.size(0), 
                        gmm_k, mu_box.size(2)), 2)).sum(-1)
        return mu_al_box, mu_ep_box

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             sampling_results,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             unc=False):
        losses = dict()
        
        cls_num = self.num_classes
        # xishu softmax    fangcha relu
        box_rep_len = cls_num * 4
        # box_rep_len = (cls_score.size(1) - 1) * 4
        gmm_k =  bbox_pred.size(1) // (box_rep_len * 3)
        bbox_pred = bbox_pred.view(bbox_pred.size(0), 3, -1)
        mu_box = bbox_pred[:, 0, :].view(bbox_pred.size(0), gmm_k, -1)
        sigma_box = bbox_pred[:, 1, :].view(bbox_pred.size(0), gmm_k, -1)
        pi_box = bbox_pred[:, 2, :].view(bbox_pred.size(0), gmm_k, -1)

        pi_box = F.softmax(pi_box, dim=1)
        sigma_box = F.sigmoid(sigma_box)

        mu_al_box = (pi_box * sigma_box).sum(dim=1).max(-1)[0]
        mu_ep_box = (pi_box * torch.pow(mu_box - (pi_box * mu_box).sum(dim=1)[:, None, :].expand(bbox_pred.size(0), 
                        gmm_k, mu_box.size(2)), 2)).sum(dim=1).max(-1)[0]

        max_mu_ep_box = (pi_box * torch.pow(mu_box - (pi_box * mu_box).sum(dim=1)[:, None, :].expand(bbox_pred.size(0), 
                        gmm_k, mu_box.size(2)), 2)).sum(dim=1).max(-1)[0]

        cls_score = cls_score.view(cls_score.size(0), gmm_k, -1)
        pi_cls = cls_score[:, :, -1:]
        mu_cls = cls_score[:, :, :cls_score.size(2) // 2]
        sigma_cls = cls_score[:, :, cls_score.size(2) // 2:-1]
        lam_cls = torch.randn(mu_cls.size()).to(mu_cls.device)

        pi_cls = F.softmax(pi_cls, dim=1)
        sigma_cls = F.sigmoid(sigma_cls)

        mu_al_cls = (pi_cls.expand(cls_score.size(0), gmm_k, sigma_cls.size(2)) * sigma_cls).sum(dim=1)
        mu_ep_cls = (pi_cls.expand(cls_score.size(0), gmm_k, sigma_cls.size(2)) * torch.pow(mu_cls - (pi_cls.expand(cls_score.size(0), gmm_k, sigma_cls.size(2)) * mu_cls).sum(dim=1)[:, None, :].expand(cls_score.size(0), 
                        gmm_k, sigma_cls.size(2)), 2)).sum(dim=1)

        if self.box_unc_type == 'al':
            box_unc_logit = mu_al_box
            box_unc_gt_static = self.box_gt_al
        elif self.box_unc_type == 'ep':
            box_unc_logit = mu_ep_box
            box_unc_gt_static = self.box_gt_ep
        
        if self.cls_unc_type == 'al':
            cls_unc_logit = mu_al_cls
            cls_unc_gt_static = self.cls_gt_al
        elif self.cls_unc_type == 'ep':
            cls_unc_logit = mu_ep_cls
            cls_unc_gt_static = self.cls_gt_ep

        # split_list = [sr.bboxes.size(0) for sr in sampling_results]
        # device = bbox_pred.device
        # batch = split_list[0]
        # pos_inds = torch.zeros([0]).to(device).long()
        # pos_gt_map = torch.zeros([0]).to(device).long()
        # # pos_label = torch.zeros([0]).to(device).long()
        # for i, res in enumerate(sampling_results):
        #     pos_inds = torch.cat([pos_inds, (torch.arange(0, res.pos_inds.size(0)).to(device).long() + (i * batch)).view(-1)])
        #     pos_gt_map = torch.cat([pos_gt_map, (res.pos_assigned_gt_inds+ (i * batch)).view(-1)])
        #     # pos_label = torch.cat([pos_label, (res.pos_gt_labels+ (i * batch)).view(-1)])
        # box_logit = box_unc_logit[pos_inds]
        # box_target = box_unc_logit[pos_gt_map]
        # map_gt_target = labels[]

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_mu_box = mu_box.view(
                        bbox_pred.size(0), gmm_k, -1,
                        4)[pos_inds.type(torch.bool), :,
                           labels[pos_inds.type(torch.bool)]]
                    pos_sigma_box = sigma_box.view(
                        bbox_pred.size(0), gmm_k, -1,
                        4)[pos_inds.type(torch.bool), :,
                           labels[pos_inds.type(torch.bool)]]
                    pos_pi_box = pi_box.view(
                        bbox_pred.size(0), gmm_k, -1,
                        4)[pos_inds.type(torch.bool), :,
                           labels[pos_inds.type(torch.bool)]]
                    max_mu_ep_box = max_mu_ep_box[pos_inds.type(torch.bool)]
                    pos_target = bbox_targets[pos_inds.type(torch.bool)]
                    pos_target = pos_target[:, None, :].expand(pos_target.size(0), gmm_k, pos_target.size(1))
                loss_box = - torch.log((pos_pi_box * self._gaussian_dist_pdf(pos_mu_box, pos_target, pos_sigma_box)).sum(1) + 1e-9).sum(-1) / self.eta
                if self.epoch > self.box_warm_epoch:
                    # print('max_mu_ep_box before softmax ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    # print(max_mu_ep_box)
                    # max_mu_ep_box = 1- F.softmax(max_mu_ep_box)
                    # losses['loss_bbox'] = ((loss_box * max_mu_ep_box).sum() / max_mu_ep_box.sum()) * self.lam_box_loss
                    # print('pos num:' + str(pos_pi_box.size(0)))
                    # print('max_mu_ep_box after softmax ----------------------------------------------------------')
                    # print(max_mu_ep_box)
                    cls_label = labels[pos_inds.type(torch.bool)]
                    box_pos_unc_gt =box_unc_gt_static[cls_label]
                    box_pos_unc = box_unc_logit[pos_inds.type(torch.bool)]
                    box_unc_reweight = abs(abs(box_pos_unc) - abs(box_pos_unc_gt))
                    box_unc_reweight = 1 - box_unc_reweight / box_unc_reweight.sum()
                    losses['loss_bbox'] = ((loss_box * box_unc_reweight).sum() / box_unc_reweight.sum()) * self.lam_box_loss
                else:
                    losses['loss_bbox'] = (loss_box).sum() / pos_mu_box.size(0)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()         
        
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                cls_score = mu_cls + (torch.sqrt(sigma_cls) * lam_cls)
                # cls_score = mu_cls

                inds = torch.arange(0, cls_score.size(0), 1)
                gt_score = cls_score[inds, :, labels[inds]]
                x_max = cls_score.max()
                loss_cls_ = pi_cls.view(pi_cls.size(0), pi_cls.size(1)) * (gt_score - (torch.log(torch.exp(cls_score - x_max).sum(dim=-1)) + x_max))

                if self.epoch > self.cls_warm_epoch:
                    loss_cls_pos = loss_cls_[pos_inds.type(torch.bool)]
                    loss_cls_neg = loss_cls_[not pos_inds.type(torch.bool)]
                    cls_label = labels[pos_inds.type(torch.bool)]
                    cls_pos_unc_gt = cls_unc_gt_static[cls_label]
                    cls_pos_unc = cls_unc_logit[pos_inds.type(torch.bool)]
                    cls_unc_reweight = abs(abs(cls_pos_unc) - abs(cls_pos_unc_gt))
                    cls_unc_reweight = 1 - cls_unc_reweight / cls_unc_reweight.sum()
                    loss_cls_pos = - ((loss_cls_pos * cls_unc_reweight).sum() / cls_unc_reweight.sum())
                    loss_cls_neg = - (loss_cls_neg.sum() / loss_cls_neg.size(0))
                    loss_cls_ = loss_cls_pos * self.cls_pos_lambda + loss_cls_neg * self.cls_neg_lambda
                else:
                    loss_cls_ = - (loss_cls_.sum() / avg_factor) * self.cls_lambda
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                Fisrt tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        cls_num = self.num_classes
        # xishu softmax    fangcha relu
        box_rep_len = cls_num * 4
        # box_rep_len = (cls_score.size(1) - 1) * 4
        gmm_k =  bbox_pred.size(1) // (box_rep_len * 3)

        cls_score = cls_score.view(cls_score.size(0), gmm_k, -1)
        pi_cls = cls_score[:, :, -1:]
        # mu_cls = cls_score[:, :, :-1]
        mu_cls = cls_score[:, :, :cls_score.size(2) // 2]
        sigma_cls = cls_score[:, :, cls_score.size(2) // 2:-1]

        pi_cls = F.softmax(pi_cls, dim=1)
        sigma_cls = F.sigmoid(sigma_cls)

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            # scores = F.softmax(
            #     cls_score, dim=-1) if cls_score is not None else None
            scores = (pi_cls * F.softmax(
                mu_cls, dim=-1)).sum(dim=1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.

        bbox_pred = bbox_pred.view(bbox_pred.size(0), 3, -1)
        mu_box = bbox_pred[:, 0, :].view(bbox_pred.size(0), gmm_k, -1)
        sigma_box = bbox_pred[:, 1, :].view(bbox_pred.size(0), gmm_k, -1)
        pi_box = bbox_pred[:, 2, :].view(bbox_pred.size(0), gmm_k, -1)

        pi_box = F.softmax(pi_box, dim=1)
        sigma_box = F.sigmoid(sigma_box)

        bbox_pred = (pi_box * mu_box).sum(dim=1)
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:

            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): Rois from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4) or
                (num_proposals, 5).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """

        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    def onnx_export(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    cfg=None,
                    **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """

        assert rois.ndim == 3, 'Only support export two stage ' \
                               'model to ONNX ' \
                               'with batch dimension. '
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class',
                                                 cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)

        scores = scores[..., :self.num_classes]
        if self.reg_class_agnostic:
            return add_dummy_nms_for_onnx(
                bboxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img)
        else:
            batch_size = scores.shape[0]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets

            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels

    def set_epoch(self, epoch):
        self.epoch = epoch

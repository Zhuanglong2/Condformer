import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import LOSSES
from mmdet.models.dab_transformer.matcher import HungarianMatcher


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, logit, target, mask=None):
        if mask is not None:
            logit = logit[mask]
            target = target[mask]
        loss = self.crit(logit, target)
        return loss


def _neg_loss(pred, gt, channel_weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if channel_weights is None:
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
    else:
        pos_loss_sum = 0
        neg_loss_sum = 0
        for i in range(len(channel_weights)):
            p = pos_loss[:, i, :, :].sum() * channel_weights[i]
            n = neg_loss[:, i, :, :].sum() * channel_weights[i]
            pos_loss_sum += p
            neg_loss_sum += n
        pos_loss = pos_loss_sum
        neg_loss = neg_loss_sum
    if num_pos > 2:
        loss = loss - (pos_loss + neg_loss) / num_pos
    else:
        loss = loss - (pos_loss + neg_loss) / 256
        loss = torch.tensor(0, dtype=torch.float32).to(pred.device)
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weights_list=None):
        return self.neg_loss(out, target, weights_list)


class RegL1KpLoss(nn.Module):

    def __init__(self):
        super(RegL1KpLoss, self).__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        mask = mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss


def compute_locations(shape, device):
    pos = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos


@LOSSES.register_module
class CondLaneLoss(torch.nn.Module):

    def __init__(self, weights, num_lane_cls):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(CondLaneLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit2 = FocalLoss()
        self.crit_class = nn.BCEWithLogitsLoss()
        self.crit_kp = RegL1KpLoss()
        self.crit_row = nn.L1Loss(reduction='none')
        self.crit_ce = nn.CrossEntropyLoss()
        self.matcher = HungarianMatcher()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.last_losses_coordinate = 0

        coordinate_weight = 0.1
        hm_weight = 1.
        class_weight = 0.1
        kps_weight = 0.4
        row_weight = 1.0
        range_weight = 1.0
        row_weight2 = 1.0

        self.hm_weight = weights[
            'hm_weight'] if 'hm_weight' in weights else hm_weight
        self.kps_weight = weights[
            'kps_weight'] if 'kps_weight' in weights else kps_weight
        self.row_weight = weights[
            'row_weight'] if 'row_weight' in weights else row_weight
        self.range_weight = weights[
            'range_weight'] if 'range_weight' in weights else range_weight
        self.coordinate_weight = weights[
            'coordinate_weight'] if 'coordinate_weight' in weights else coordinate_weight
        self.class_weight = weights[
            'class_weight'] if 'class_weight' in weights else class_weight
        self.row_weight2 = weights[
            'row_weight2'] if 'row_weight2' in weights else row_weight2

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_coordinate(self, outputs, outputs_class, Targets, indices):
        if indices is not None:
            targets = Targets
            idx = self._get_src_permutation_idx(indices)

            src_lines = outputs[idx]
            target_lines = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
            # loss_lines = self.crit_kp(src_lines, target_lines, target_lines2)
            loss_lines = F.l1_loss(src_lines, target_lines, reduction='none')
            # losses_coordinate = loss_lines.sum() / num_boxes
            losses_coordinate = loss_lines.sum()

            # 分类loss(正负样本)
            target_classes = torch.zeros(outputs_class.shape[:-1], dtype=torch.int64, device=outputs_class.device)
            target_classes[idx] = 1

            outputs_class = torch.cat([v for v in outputs_class])
            target_classes = torch.cat([v for v in target_classes])

            loss_class = self.CrossEntropyLoss(outputs_class, target_classes)
            # losses_class = loss_class.sum() / num_boxes
            losses_class = loss_class.sum()
        else:#全是负样本
            # 分类loss(正负样本)
            target_classes = torch.zeros(outputs_class.shape[:-1], dtype=torch.int64, device=outputs_class.device)

            outputs_class = torch.cat([v for v in outputs_class])
            target_classes = torch.cat([v for v in target_classes])

            loss_class = self.CrossEntropyLoss(outputs_class, target_classes)
            # losses_class = loss_class.sum() / num_boxes
            losses_class = loss_class.sum()
            losses_coordinate = losses_class
        losses = {
            "losses_coordinate": losses_coordinate,
            "loss_class": losses_class,
        }

        return losses

    def loss_class(self, outputs_class, indices):
        if indices is not None:
            idx = self._get_src_permutation_idx(indices)

            # 分类loss(正负样本)
            target_classes = torch.zeros(outputs_class.shape[:-1], dtype=torch.int64, device=outputs_class.device)
            target_classes[idx] = 1

            outputs_class = torch.cat([v for v in outputs_class])
            target_classes = torch.cat([v for v in target_classes])

            loss_class = self.CrossEntropyLoss(outputs_class, target_classes)
            # losses_class = loss_class.sum() / num_boxes
            losses_class = loss_class.sum()
        else:  # 全是负样本
            # 分类loss(正负样本)
            target_classes = torch.zeros(outputs_class.shape[:-1], dtype=torch.int64, device=outputs_class.device)

            outputs_class = torch.cat([v for v in outputs_class])
            target_classes = torch.cat([v for v in target_classes])

            loss_class = self.CrossEntropyLoss(outputs_class, target_classes)
            # losses_class = loss_class.sum() / num_boxes
            losses_class = loss_class.sum()

        losses = {
            "loss_class": losses_class,
        }

        return losses

    def forward(self, poses, output, meta, **kwargs):
        mask_class, kps, mask, lane_range = output[:4]
        # hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        class_loss, coordinate_loss, hm_loss, hm2_loss, kps_loss, row_loss, range_loss, row_loss2 = 0, 0, 0, 0, 0, 0, 0, 0

        # index = self.matcher(params2, params_label)
        # loss = self.loss_coordinate(params2, params2_class, params_label, index)
        # idx = self._get_src_permutation_idx(index)

        # if self.coordinate_weight > 0:
        #     coordinate_loss += loss['losses_coordinate']
        #

        # if self.hm_weight > 0:
        #     hm_loss += self.crit(hm, kwargs['gt_hm'])

        # matcher_mask = torch.clamp(mask.sigmoid_(), min=0, max=1)
        if mask.shape[2] == 40:
            index = self.matcher(mask.view(1, -1, 4000), kwargs['gt_rows2'].view(1, -1, 4000))
        else:
            index = self.matcher(mask.view(1, -1, 16000), kwargs['gt_rows2'].view(1, -1, 16000))
        idx = self._get_src_permutation_idx(index)

        if self.row_weight2 > 0:
            mask2 = mask[idx].unsqueeze(0)
            gt_rows2 = torch.cat([t[i] for t, (_, i) in zip(kwargs['gt_rows2'], index)], dim=0)
            row_loss2 += self.crit_row(mask2, gt_rows2.unsqueeze(0))  # 1 10 40 100

        if self.class_weight > 0:
            loss = self.loss_class(mask_class, index)
            class_loss += loss['loss_class']

        if self.kps_weight > 0:
            kps = kps[idx].unsqueeze(0)
            gt_reg = torch.cat([t[i] for t, (_, i) in zip(kwargs['gt_reg'], index)], dim=0)
            gt_reg_mask = torch.cat([t[i] for t, (_, i) in zip(kwargs['gt_reg_mask'], index)], dim=0)
            kps_loss += self.crit_kp(kps, gt_reg.unsqueeze(0), gt_reg_mask.unsqueeze(0))  # 1 10 40 100
            # kps_loss += self.crit_kp(kps, kwargs['gt_reg'], kwargs['gt_reg_mask'])#1 10 40 100

        if self.row_weight > 0:
            mask = mask[idx].unsqueeze(0)
            mask_softmax = F.softmax(mask, dim=3)
            pos = compute_locations(mask_softmax.size(), device=mask_softmax.device)
            row_pos = torch.sum(pos * mask_softmax, dim=3) + 0.5
            gt_rows = torch.cat([t[i] for t, (_, i) in zip(kwargs['gt_rows'], index)], dim=0)
            gt_row_masks = torch.cat([t[i] for t, (_, i) in zip(kwargs['gt_row_masks'], index)], dim=0)
            row_loss += self.crit_kp(row_pos, gt_rows.unsqueeze(0), gt_row_masks.unsqueeze(0))
            # row_loss += self.crit_kp(row_pos, kwargs['gt_rows'], kwargs['gt_row_masks'])

        if self.range_weight > 0:
            lane_range = lane_range.unsqueeze(0)[idx]
            gt_ranges = torch.cat([t[i] for t, (_, i) in zip(kwargs['gt_ranges'].unsqueeze(0), index)], dim=0)
            range_loss = self.crit_ce(lane_range, gt_ranges)
            # range_loss = self.crit_ce(lane_range, kwargs['gt_ranges'])

        # Only non-zero losses are valid, otherwise multi-GPU training will report an error
        losses = {}
        # if self.hm_weight:
        #     losses['hm_loss'] = self.hm_weight * hm_loss
        if self.class_weight > 0:
            losses['class_loss'] = self.class_weight * class_loss
        # if self.coordinate_weight > 0:
        #     losses['coordinate_loss'] = self.coordinate_weight * coordinate_loss
        if self.kps_weight:
            losses['kps_loss'] = self.kps_weight * kps_loss
        if self.row_weight > 0:
            losses['row_loss'] = self.row_weight * row_loss
        if self.row_weight2 > 0:
            losses['row_loss2'] = self.row_weight2 * row_loss2
        if self.range_weight > 0:
            losses['range_loss'] = self.range_weight * range_loss

        return losses


@LOSSES.register_module
class CondLaneRNNLoss(CondLaneLoss):
    """for curvelanes rnn"""

    def __init__(self, weights, num_lane_cls):
        super(CondLaneRNNLoss, self).__init__(weights, num_lane_cls)
        state_weight = 1.0
        self.state_weight = weights[
            'state_weight'] if 'state_weight' in weights else state_weight

    def forward(self, output, meta, **kwargs):
        hm, kps, mask, lane_range, states = output[:5]
        hm_loss, kps_loss, row_loss, range_loss, state_loss = 0, 0, 0, 0, 0
        hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        # losses for lane, seg_loss, ins_loss, kp_loss
        if self.hm_weight > 0:
            hm_loss += self.crit(hm, kwargs['gt_hm'])
        if self.kps_weight > 0:
            kps_loss += self.crit_kp(kps, kwargs['gt_reg'],
                                     kwargs['gt_reg_mask'])
        if self.state_weight > 0:
            state_loss += self.crit_ce(states, kwargs['gt_states'])
        if self.row_weight > 0:
            mask_softmax = F.softmax(mask, dim=3)
            pos = compute_locations(
                mask_softmax.size(), device=mask_softmax.device)
            row_pos = torch.sum(pos * mask_softmax, dim=3)
            row_loss += self.crit_kp(row_pos, kwargs['gt_rows'],
                                     kwargs['gt_row_masks'])
        if self.range_weight > 0:
            range_loss = self.crit_ce(lane_range, kwargs['gt_ranges'])

        losses = {}
        if self.hm_weight:
            losses['hm_loss'] = self.hm_weight * hm_loss
        if self.kps_weight:
            losses['kps_loss'] = self.kps_weight * kps_loss
        if self.row_weight > 0:
            losses['row_loss'] = self.row_weight * row_loss
        if self.state_weight > 0:
            losses['state_loss'] = self.state_weight * state_loss
        if self.range_weight > 0:
            losses['range_loss'] = self.range_weight * range_loss
        return losses

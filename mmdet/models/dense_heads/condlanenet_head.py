import math

import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch import nn
import torch.functional as F
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .ctnet_head import CtnetHead, CtnetHead_mid
from .conv_rnn import CLSTM_cell
from ..dab_transformer.matcher import build_matcher
from ..transformer import build_transformer, build_position_encoding
from ..dab_transformer import build_transformer2, build_position_encoding2

from ..transformer.matcher import HungarianMatcher
from ..vision_transformer.vit_model import VisionTransformer
from ...ops.deform_conv_v2 import DeformConv2d

from fvcore.nn.weight_init import c2_msra_fill
from torch.nn import init

def \
        parse_dynamic_params(params,
                             channels,
                             weight_nums,
                             bias_nums,
                             out_channels=1,
                             mask=True):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    # params: (num_ins x n_param)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]
    if mask:
        bias_splits[-1] = bias_splits[-1] - 2.19

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * out_channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * out_channels)

    return weight_splits, bias_splits


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class DynamicMaskHead(nn.Module):

    def __init__(self,
                 num_layers,
                 channels,
                 in_channels,
                 mask_out_stride,
                 weight_nums,
                 bias_nums,
                 disable_coords=False,
                 shape=(160, 256),
                 out_channels=1,
                 compute_locations_pre=True,
                 location_configs=None):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.in_channels = in_channels
        self.mask_out_stride = mask_out_stride
        self.disable_coords = disable_coords

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.out_channels = out_channels
        self.compute_locations_pre = compute_locations_pre
        self.location_configs = location_configs

        if compute_locations_pre and location_configs is not None:
            N, _, H, W = location_configs['size']
            device = location_configs['device']
            locations = compute_locations(H, W, stride=1, device='cpu')

            locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, H, W)
            locations[:0, :, :] /= H
            locations[:1, :, :] /= W
            locations = locations.repeat(N, 1, 1, 1)
            self.locations = locations.to(device)

    def forward(self, x, mask_head_params, num_ins, is_mask=True):

        N, _, H, W = x.size()
        if not self.disable_coords:
            if self.compute_locations_pre and self.location_configs is not None:
                locations = self.locations.to(x.device)
            else:
                locations = compute_locations(x.size(2), x.size(3), stride=1, device='cpu')
                locations = locations.unsqueeze(0).permute(0, 2, 1).contiguous().float().view(1, 2, H, W)
                locations[:0, :, :] /= H
                locations[:1, :, :] /= W
                locations = locations.repeat(N, 1, 1, 1)
                locations = locations.to(x.device)

            # relative_coords = relative_coords.to(dtype=mask_feats.dtype)
            x = torch.cat([locations, x], dim=1)

        mask_head_inputs = []
        for idx in range(N):
            mask_head_inputs.append(x[idx:idx + 1, ...].repeat(1, num_ins[idx], 1, 1))
        mask_head_inputs = torch.cat(mask_head_inputs, 1)
        num_insts = sum(num_ins)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = parse_dynamic_params(
            mask_head_params,  # (50 67)
            self.channels,  # 64
            self.weight_nums,  # 66
            self.bias_nums,  # 1
            out_channels=self.out_channels,  # 1
            mask=is_mask)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, num_insts)
        mask_logits = mask_logits.view(1, -1, H, W)
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # x = self.def_conv(x, w, b, num_insts)
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)  #
            if i < n_layers - 1:
                x = F.relu(x)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, 1)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class MLP2(nn.Module):
    # Modified from facebookresearch/detr
    # Very simple multi-layer perceptron (also called FFN)

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

class InstanceBranch(nn.Module):

    def __init__(self, dim = 134, num_convs = 4, num_masks = 4,kernel_dim=128,num_classes=2, in_channels = None):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM

        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.mask_kernel2 = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def format(self, inds):
        ret = []
        i = 0
        for y, x, c in zip(inds[0], inds[1], inds[2]):
            if i == 1:
                break
            id_class = c + 1
            coord = [x, y]
            ret.append({
                'coord': coord,
                'id_class': id_class
            })
            i += 1
        return ret
    def parse_pos(self, seeds, h, w, device):
        pos_list = [[p['coord'], p['id_class'] - 1] for p in seeds]
        poses = []
        for p in pos_list:
            [c, r], label = p
            pos = label * h * w + r * w + c
            poses.append(pos)
        poses = torch.from_numpy(np.array(
            poses, np.long)).long().to(device).unsqueeze(1)
        return poses

    def get_seeds(self, iam_prob):
        iam_prob = torch.cat([v for v in iam_prob])
        seeds = []
        for i in range(iam_prob.shape[0]):
            heat_map = iam_prob[i].unsqueeze(0).permute(1, 2, 0).detach().cpu().numpy()
            max = np.max(heat_map)
            inds = np.where(heat_map == max)  # 返回大于threshold的索引
            seeds += self.format(inds)
        return seeds

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        params = features
        params = params.permute(0, 2, 3, 1).contiguous().view(-1, 134)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)

        seeds = self.get_seeds(iam_prob)
        pos_tensor = self.parse_pos(seeds, 40, 100, iam_prob.device)
        pos_tensor = pos_tensor.expand(-1, 134)
        mask_pos_tensor = pos_tensor[:, :67]
        reg_pos_tensor = pos_tensor[:, 67:]
        mask_params = params[:, :67].gather(0, mask_pos_tensor)
        reg_params = params[:, 67:].gather(0, reg_pos_tensor)

        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        # pred_kernel2 = self.mask_kernel2(inst_features)
        # pred_scores = self.objectness(inst_features)
        return pred_logits, mask_params, reg_params, pred_kernel, iam

class InstanceBranch2(nn.Module):

    def __init__(self, dim = 256, num_convs = 4, num_masks = 4,kernel_dim=128,num_classes=2, in_channels = None):
        super().__init__()
        # norm = cfg.MODEL.SPARSE_INST.DECODER.NORM

        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        # pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        # pred_scores = self.objectness(inst_features)
        return pred_kernel, iam

def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)

class MaskBranch(nn.Module):

    def __init__(self, dim = 256, num_convs = 4,kernel_dim=128, in_channels = None):
        super().__init__()

        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)

class BaseIAMDecoder(nn.Module):

    def __init__(self, in_channels = 256, scale_factor = 2.0, output_iam = False):
        super().__init__()
        # add 2 for coordinates
        in_channels = in_channels + 2

        self.scale_factor = scale_factor
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(in_channels = in_channels)
        self.mask_branch = MaskBranch(in_channels = in_channels)
        self.reg_branch = MaskBranch(in_channels=in_channels)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features) # 4 2 40 100
        features = torch.cat([coord_features, features], dim=1)
        pred_logits, mask_params, reg_params, pred_kernel, iam = self.inst_branch(features)
        mask_features = self.mask_branch(features)
        # reg_features = self.reg_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(B, C, H * W)).view(B, N, H, W)
        # pred_regs = torch.bmm(pred_kernel2, reg_features.view(B, C, H * W)).view(B, N, H, W)

        # pred_masks = F.interpolate(
        #     pred_masks, scale_factor=self.scale_factor,
        #     mode='bilinear', align_corners=False)
        pred_logits = torch.cat([v for v in pred_logits])
        pred_masks = torch.cat([v for v in pred_masks])
        # pred_regs = torch.cat([v for v in pred_regs])
        output = {
            "pred_logits": pred_logits,
            "mask_params": mask_params,
            "reg_params": reg_params,
            "pred_masks": pred_masks
        }

        if self.output_iam:
            iam = F.interpolate(iam, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
            output['pred_iam'] = iam

        return output

class BaseIAMDecoder2(nn.Module):

    def __init__(self, in_channels = 256, scale_factor = 2.0, output_iam = False):
        super().__init__()
        # add 2 for coordinates
        in_channels = in_channels + 2

        self.scale_factor = scale_factor
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch2(in_channels = in_channels)
        self.mask_branch = MaskBranch(in_channels = in_channels)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features) # 4 2 40 100
        features = torch.cat([coord_features, features], dim=1)
        pred_kernel, iam = self.inst_branch(features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(B, C, H * W)).view(B, N, H, W)

        # pred_masks = F.interpolate(
        #     pred_masks, scale_factor=self.scale_factor,
        #     mode='bilinear', align_corners=False)
        pred_masks = torch.cat([v for v in pred_masks])
        output = {
            "pred_masks": pred_masks
        }

        if self.output_iam:
            iam = F.interpolate(iam, scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)
            output['pred_iam'] = iam

        return output
@HEADS.register_module
class CondLaneHead(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 batch_size,

                 expansion=1,  # Expansion rate (1x for TuSimple & 2x for CULane)
                 num_queries=5,  # Maximum number of lanes
                 pos_type='sine',
                 num_heads=8,
                 drop_out=0.1,
                 enc_layers=6,
                 dec_layers=6,
                 pre_norm=False,
                 return_intermediate=True,
                 query_dim=256,
                 dim_feedforward = 2048,

                 thresh=0.5,

                 head_channels=64,
                 head_layers=1,
                 disable_coords=False,
                 branch_in_channels=288,
                 branch_channels=64,
                 branch_out_channels=64,  # 输出改成para维度
                 reg_branch_channels=32,
                 branch_num_conv=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 hm_idx=-1,
                 mask_idx=0,
                 compute_locations_pre=True,
                 location_configs=None,
                 mask_norm_act=True,
                 regression=True,
                 train_cfg=None,
                 test_cfg=None):
        super(CondLaneHead, self).__init__()
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.mask_idx = mask_idx
        self.regression = regression
        self.num_queries = num_queries
        if mask_norm_act:
            final_norm_cfg = dict(type='BN', requires_grad=True)
            final_act_cfg = dict(type='ReLU')
        else:
            final_norm_cfg = None
            final_act_cfg = None
        # mask branch
        mask_branch = []
        mask_branch.append(
            ConvModule(
                sum(in_channels),
                branch_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg))
        for i in range(branch_num_conv):
            mask_branch.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg))
        mask_branch.append(
            ConvModule(
                branch_channels,
                branch_out_channels,
                kernel_size=3,
                padding=1,
                # stride=2,
                norm_cfg=final_norm_cfg,
                act_cfg=final_act_cfg))
        self.add_module('mask_branch', nn.Sequential(*mask_branch))


        self.mask_weight_nums, self.mask_bias_nums = self.cal_num_params(
            head_layers, disable_coords, head_channels, out_channels=1)

        self.num_mask_params = sum(self.mask_weight_nums) + sum(
            self.mask_bias_nums)

        self.reg_weight_nums, self.reg_bias_nums = self.cal_num_params(
            head_layers, disable_coords, head_channels, out_channels=1)

        self.num_reg_params = sum(self.reg_weight_nums) + sum(
            self.reg_bias_nums)
        if self.regression:
            self.num_gen_params = self.num_mask_params + self.num_reg_params
        else:
            self.num_gen_params = self.num_mask_params
            self.num_reg_params = 0

        self.mask_head = DynamicMaskHead(
            head_layers,
            branch_out_channels,
            branch_out_channels,
            1,
            self.mask_weight_nums,
            self.mask_bias_nums,
            disable_coords=False,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        if self.regression:
            self.reg_head = DynamicMaskHead(
                head_layers,
                branch_out_channels,
                branch_out_channels,
                1,
                self.reg_weight_nums,
                self.reg_bias_nums,
                disable_coords=False,
                out_channels=1,
                compute_locations_pre=compute_locations_pre,
                location_configs=location_configs)
        if 'params' not in heads:
            # heads['params'] = num_classes * (self.num_reg_params)
            heads['params'] = num_classes * (self.num_mask_params + self.num_reg_params)

        self.ctnet_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            # head_conv=64,)
            head_conv=branch_in_channels)

        self.BaseIAMDecoder = BaseIAMDecoder()

        self.feat_width = location_configs['size'][-1]
        self.mlp = MLP(self.feat_width, 64, 2, 2)

        # dab_transformer
        hidden_dim = 134 * expansion
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.position_embedding = build_position_encoding2(hidden_dim=query_dim, position_embedding=pos_type)
        self.input_proj = nn.Conv2d(64 * expansion, query_dim, kernel_size=1)
        # self.input_proj = nn.Conv2d(64 * expansion, 7, kernel_size=1)
        self.transformer = build_transformer2(hidden_dim=query_dim,
                                              query_dim=hidden_dim,
                                              dropout=drop_out,
                                              nheads=num_heads,
                                              dim_feedforward=dim_feedforward,
                                              enc_layers=enc_layers,
                                              dec_layers=dec_layers,
                                              pre_norm=pre_norm,
                                              return_intermediate_dec=return_intermediate)
        self.class_embed = nn.Linear(query_dim, 2)
        self.para_embed = MLP2(query_dim, query_dim, 134, 3)
        # self.class_embed = nn.Linear(250, 2)
        # self.para_embed = MLP2(250, 250, 134, 3)
        self.query_dim = hidden_dim
        self.thresh = thresh

        self.num_ins = []
        for i in range(batch_size):
            self.num_ins.append(self.num_queries)

    def cal_num_params(self,
                       num_layers,
                       disable_coords,
                       channels,
                       out_channels=1):
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    weight_nums.append((channels + 2) * out_channels)
                else:
                    weight_nums.append(channels * out_channels)
                bias_nums.append(out_channels)
            elif l == 0:
                if not disable_coords:
                    weight_nums.append((channels + 2) * channels)
                else:
                    weight_nums.append(channels * channels)
                bias_nums.append(channels)

            else:
                weight_nums.append(channels * channels)
                bias_nums.append(channels)
        return weight_nums, bias_nums

    def ctdet_decode(self, heat, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2  # 计算padding

            hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)  # 最大值池化
            keep = (hmax == heat).float()  # 未被池化的给1
            return heat * keep  # 只留下最大值部分

        def _format(heat, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,
                    'id_class': id_class,
                    'score': score
                })
            return ret

        # 1 1 20 50
        heat_nms = _nms(heat)  # 1 1 20 50
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # 20 50 1
        inds = np.where(heat_nms > thr)  # 返回大于threshold的索引
        seeds = _format(heat_nms, inds)

        # if len(seeds) < results_num_class:
        #     inds2 = np.where(heat_nms > 0)
        #     seeds = _format(heat_nms, inds2)
        #     t2 = sorted(seeds, key=lambda x: x['score'], reverse=True)
        #     Seeds = t2[:results_num_class]
        # else:
        #     Seeds = seeds

        # append
        # if len(seeds) < results_num_class:
        #     seeds += self.last_seeds
        #     # if len(self.last_seeds) >= results_num_class:
        #     #     seeds = self.last_seeds #用上次的点
        #     # else:
        #     #     seeds += self.last_seeds
        #
        # if len(seeds) != 0:
        #     self.last_seeds = seeds

        return seeds

    def lane_pruning(self, existence, existence_conf, max_lane):
        # Prune lanes based on confidence (a max number constrain for lanes in an image)
        # Maybe too slow (but should be faster than topk/sort),
        # consider batch size >> max number of lanes
        while (existence.sum(dim=1) > max_lane).sum() > 0:
            indices = (existence.sum(dim=1, keepdim=True) > max_lane).expand_as(existence) * \
                      (existence_conf == existence_conf.min(dim=1, keepdim=True).values)
            existence[indices] = 0
            existence_conf[indices] = 1.1  # So we can keep using min

        return existence, existence_conf

    def result(self, outputs_class, output_specific):

        existence_conf = outputs_class.sigmoid()[..., -1]
        # topk_values, topk_indexes = torch.topk(prob.view(outputs_class.shape[0], -1), num_select, dim=1)
        existence = existence_conf > self.thresh
        # existence, _ = self.lane_pruning(existence, existence_conf, max_lane=4)#作用?
        # coordinate
        Coordinates = []
        for i in range(existence.shape[1]):
            if existence[0, i]:
                Coordinates.append({
                    'coord': [int(output_specific[0, i, 0] * 50), int(output_specific[0, i, 1] * 20)],
                    'score': existence_conf[0, i],
                    'id_class': 1,
                })

        return Coordinates


    def result2(self, outputs_params, outputs_class, thresh):

        existence_conf = torch.softmax(outputs_class, dim=2)[..., -1]
        # topk_values, topk_indexes = torch.topk(existence_conf, len(seeds), dim=1)
        existence = existence_conf > thresh
        # coordinate
        num_ins = []
        conut = 0
        params = None
        seeds = []

        for i in range(existence.shape[1]):
            if existence[0, i]:
                if conut == 0:
                    params = outputs_params[i].unsqueeze(0)
                else:
                    params = torch.cat((params, outputs_params[i].unsqueeze(0)), dim=0)
                conut += 1
                seeds.append({'reg': 0})
        num_ins.append(conut)

        return params, num_ins, seeds

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, center, radius):
        b = len(center)
        heatmap = np.zeros((b, 1, 20, 50), np.float32)
        k = 1
        # radius = 2
        for i in range(b):
            for j in range(len(center[i])):
                Center = center[i][j]['coord']
                diameter = 2 * radius + 1
                gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
                x, y = int(Center[0]), int(Center[1])
                height, width = heatmap[i, 0].shape[0:2]
                left, right = min(x, radius), min(width - x, radius + 1)
                top, bottom = min(y, radius), min(height - y, radius + 1)
                masked_heatmap = heatmap[i, 0][y - top:y + bottom, x - left:x + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
                    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        heatmap = torch.from_numpy(heatmap)
        # show
        # plt.imshow(heatmap[0, 0], origin='lower', aspect='auto', cmap='binary')
        # plt.show()

        return heatmap

    def fusion_pm(self, pm, last_pm):
        if len(last_pm) != 0:
            c = torch.cat((pm, last_pm), dim=1)
            out = self.fc_PARAMETER(c)
            self.last_pm = out.detach()
        else:
            out = pm
            self.last_pm = out.detach()
        return out

    def inverse_sigmoid(self, x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def NMS(self, seeds1, seeds2, threshold = 8):
        # 删除与hm得到的坐标重复
        seeds = []
        state = 1
        for i in range(len(seeds1)):
            for j in range(len(seeds2)):
                difference = abs(seeds1[i]['coord'][0] - seeds2[j]['coord'][0]) + abs(seeds1[i]['coord'][1] - seeds2[j]['coord'][1])
                if difference < threshold:
                    state = 0
                    continue
            if state:
                seeds.append({
                    'coord': seeds1[i]['coord'],
                    'id_class': 1,
                    'score': seeds1[i]['score'],
                    })
            state = 1

        seeds += seeds2
        return seeds

    def forward_train(self, inputs, pose, num_ins, hm_thr=0.5):
        x_list = list(inputs)
        f_hm = x_list[2]
        f_mask = x_list[self.mask_idx]

        m_batchsize = f_hm.size()[0]
        h_mask, w_mask = f_mask.size()[2:]

        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch

        #ablation
        main_branch = self.input_proj(f_hm)
        for i in range(m_batchsize):
            main_branch_ = main_branch[i].unsqueeze(0)
            padding_masks = torch.zeros((main_branch_.shape[0], main_branch_.shape[2], main_branch_.shape[3]), dtype=torch.bool, device=main_branch_.device)
            Pos = self.position_embedding(main_branch_, padding_masks)  # 4 4 20 50
            hs, reference = self.transformer(main_branch_, padding_masks, self.query_embed.weight, Pos)
            if i == 0:
                tmp = self.para_embed(hs)
                tmp[..., :self.query_dim] += reference
                tmp_params = tmp[-1]
                tmp_class = hs[-1]
            else:
                tmp = self.para_embed(hs)
                tmp[..., :self.query_dim] += reference
                tmp_params = torch.cat((tmp_params, tmp[-1]), dim=1)
                tmp_class = torch.cat((tmp_class, hs[-1]), dim=1)
        outputs_class = self.class_embed(tmp_class)
        params = tmp_params[0]

        # main_branch = self.input_proj(f_hm).reshape(4, 7, -1)
        # outputs_class = self.class_embed(main_branch).unsqueeze(0).reshape(1, -1, 2)
        # params = self.para_embed(main_branch).reshape(-1, 134)
        ####
        mask_params = params[:, :self.num_mask_params]
        masks = self.mask_head(mask_branch, mask_params, self.num_ins)

        if self.regression:
            reg_params = params[:, self.num_mask_params:]
            regs = self.reg_head(reg_branch, reg_params, self.num_ins)
        else:
            regs = masks

        feat_range = masks.permute(0, 1, 3, 2).view(sum(self.num_ins), w_mask, h_mask)
        feat_range = self.mlp(feat_range)

        return outputs_class, regs, masks, feat_range, [mask_branch, reg_branch]

    def forward_test(
            self,
            inputs,
            hack_seeds=None,
            hm_thr=0.3,
    ):

        def parse_pos(seeds, batchsize, num_classes, h, w, device):
            pos_list = [[p['coord'], p['id_class'] - 1] for p in seeds]
            poses = []
            for p in pos_list:
                [c, r], label = p
                pos = label * h * w + r * w + c
                poses.append(pos)
            poses = torch.from_numpy(np.array(poses, np.long)).long().to(device).unsqueeze(1)
            # pos_list = [[p['coord']] for p in seeds]
            # poses = []
            # for p in pos_list:
            #     [c, r] = p[0]
            #     pos = r * w + c
            #     poses.append(pos)
            # poses = torch.from_numpy(np.array(poses, np.long)).long().to(device).unsqueeze(1)
            return poses

        x_list = list(inputs)
        f_hm = x_list[2]
        f_mask = x_list[self.mask_idx]

        h_mask, w_mask = f_mask.size()[2:]

        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch

        main_branch = self.input_proj(f_hm)
        padding_masks = torch.zeros((main_branch.shape[0], main_branch.shape[2], main_branch.shape[3]), dtype=torch.bool, device=main_branch.device)
        Pos = self.position_embedding(main_branch, padding_masks)  # 4 4 20 50
        hs, reference = self.transformer(main_branch, padding_masks, self.query_embed.weight, Pos)
        tmp = self.para_embed(hs)
        tmp[..., :self.query_dim] += reference
        tmp_params = tmp[-1]
        tmp_class = hs[-1]
        outputs_class = self.class_embed(tmp_class)
        params = tmp_params[0]

        # main_branch = self.input_proj(f_hm).reshape(1, 7, -1)
        # outputs_class = self.class_embed(main_branch).unsqueeze(0).reshape(1, -1, 2)
        # params = self.para_embed(main_branch).reshape(-1, 134)

        params, num_ins, seeds = self.result2(params, outputs_class, self.thresh)

        if num_ins[0] == 0:
            return [], []
        else:
            mask_params = params[:, :self.num_mask_params]
            masks = self.mask_head(mask_branch, mask_params, num_ins)

            if self.regression:
                reg_params = params[:, self.num_mask_params:]
                regs = self.reg_head(reg_branch, reg_params, num_ins)
            else:
                regs = masks

            feat_range = masks.permute(0, 1, 3, 2).view(sum(num_ins), w_mask, h_mask)
            feat_range = self.mlp(feat_range)

            for i in range(num_ins[0]):
                seeds[i]['reg'] = regs[0, i:i + 1, :, :]#1 1 40 100
                m = masks[0, i:i + 1, :, :]
                seeds[i]['mask'] = m #1 40 100
                seeds[i]['range'] = feat_range[i:i + 1]#2 40

            return seeds, masks

    def inference_mask(self, pos):
        pass

    def forward(
            self,
            x_list,
            hm_thr=0.3,
    ):
        return self.forward_test(x_list, )

    def init_weights(self):
        # ctnet_head will init weights during building
        pass


class PredictFC(nn.Module):

    def __init__(self, num_params, num_states, in_channels):
        super(PredictFC, self).__init__()
        self.num_params = num_params
        self.fc_param = nn.Conv2d(
            in_channels,
            num_params,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc_state = nn.Conv2d(
            in_channels,
            num_states,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, input):
        params = self.fc_param(input)
        state = self.fc_state(input)
        return params, state


@HEADS.register_module
class CondLaneRNNHead(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 ct_head,
                 head_channels=64,
                 head_layers=1,
                 disable_coords=False,
                 branch_channels=64,
                 branch_out_channels=64,
                 reg_branch_channels=32,
                 branch_num_conv=1,
                 num_params=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 hm_idx=-1,
                 mask_idx=0,
                 compute_locations_pre=True,
                 location_configs=None,
                 zero_hidden_state=False,
                 train_cfg=None,
                 test_cfg=None):
        super(CondLaneRNNHead, self).__init__()
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.mask_idx = mask_idx
        self.zero_hidden_state = zero_hidden_state

        # mask branch
        mask_branch = []
        mask_branch.append(
            ConvModule(
                sum(in_channels),
                branch_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg))
        for i in range(branch_num_conv):
            mask_branch.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg))
        mask_branch.append(
            ConvModule(
                branch_channels,
                branch_out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=None,
                act_cfg=None))
        self.add_module('mask_branch', nn.Sequential(*mask_branch))

        self.mask_weight_nums, self.mask_bias_nums = self.cal_num_params(
            head_layers, disable_coords, branch_out_channels, out_channels=1)

        self.num_mask_params = sum(self.mask_weight_nums) + sum(
            self.mask_bias_nums)

        self.reg_weight_nums, self.reg_bias_nums = self.cal_num_params(
            head_layers, disable_coords, reg_branch_channels, out_channels=1)

        self.num_reg_params = sum(self.reg_weight_nums) + sum(
            self.reg_bias_nums)
        self.num_gen_params = self.num_mask_params + self.num_reg_params

        self.mask_head = DynamicMaskHead(
            head_layers,
            branch_out_channels,
            branch_out_channels,
            1,
            self.mask_weight_nums,
            self.mask_bias_nums,
            disable_coords=False,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        self.reg_head = DynamicMaskHead(
            head_layers,
            reg_branch_channels,
            reg_branch_channels,
            1,
            self.reg_weight_nums,
            self.reg_bias_nums,
            disable_coords=False,
            out_channels=1,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        self.ctnet_head = CtnetHead(
            ct_head['heads'],
            channels_in=ct_head['channels_in'],
            final_kernel=ct_head['final_kernel'],
            head_conv=ct_head['head_conv'])
        self.rnn_in_channels = ct_head['heads']['params']
        self.rnn_ceil = CLSTM_cell((1, 1), self.rnn_in_channels, 1,
                                   self.rnn_in_channels)
        self.final_fc = PredictFC(self.num_gen_params, 2, self.rnn_in_channels)
        self.feat_width = location_configs['size'][-1]
        self.mlp = MLP(self.feat_width, 64, 2, 2)

    def cal_num_params(self,
                       num_layers,
                       disable_coords,
                       channels,
                       out_channels=1):
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    weight_nums.append((channels + 2) * out_channels)
                else:
                    weight_nums.append(channels * out_channels)
                bias_nums.append(out_channels)
            elif l == 0:
                if not disable_coords:
                    weight_nums.append((channels + 2) * channels)
                else:
                    weight_nums.append(channels * channels)
                bias_nums.append(channels)

            else:
                weight_nums.append(channels * channels)
                bias_nums.append(channels)
        return weight_nums, bias_nums

    def ctdet_decode(self, heat, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        def _format(heat, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,
                    'id_class': id_class,
                    'score': score
                })
            return ret

        heat_nms = _nms(heat)
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        return seeds

    def forward_train(self, inputs, pos, num_ins, memory):

        def choose_idx(num_ins, idx):
            count = 0
            for i in range(len(num_ins) - 1):
                if idx >= count and idx < count + num_ins[i]:
                    return i
                count += num_ins[i]
            return len(num_ins) - 1

        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]

        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]

        # f_mask
        z = self.ctnet_head(f_hm)
        hm, params = z['hm'], z['params']
        h_hm, w_hm = hm.size()[2:]
        h_mask, w_mask = f_mask.size()[2:]
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)
        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        params = params.permute(0, 1, 3, 4,
                                2).contiguous().view(-1, self.rnn_in_channels)
        pos_array = np.array([p[0] for p in pos], np.int32)
        pos_tensor = torch.from_numpy(pos_array).long().to(
            params.device).unsqueeze(1)

        pos_tensor = pos_tensor.expand(-1, self.rnn_in_channels)
        states = []
        kernel_params = []
        if pos_tensor.size()[0] == 0:
            masks = None
            regs = None
        else:
            num_ins_per_seed = []
            rnn_params = params.gather(0, pos_tensor)
            ins_count = 0
            for idx, (_, r_times) in enumerate(pos):
                rnn_feat_input = rnn_params[idx:idx + 1, :]
                rnn_feat_input = rnn_feat_input.reshape(1, -1, 1, 1)
                hidden_h = rnn_feat_input
                hidden_c = rnn_feat_input
                rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)
                if self.zero_hidden_state:
                    hidden_state = None
                else:
                    hidden_state = (hidden_h, hidden_c)
                num_ins_count = 0
                for _ in range(r_times):
                    rnn_out, hidden_state = self.rnn_ceil(
                        inputs=rnn_feat_input,
                        hidden_state=hidden_state,
                        seq_len=1)
                    rnn_out = rnn_out.reshape(1, -1, 1, 1)
                    k_param, state = self.final_fc(rnn_out)
                    k_param = k_param.squeeze(-1).squeeze(-1)
                    state = state.squeeze(-1).squeeze(-1)
                    states.append(state)
                    kernel_params.append(k_param)
                    num_ins_count += 1
                    rnn_feat_input = rnn_out
                    rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)
                    ins_count += 1

                num_ins_per_seed.append(num_ins_count)
            kernel_params = torch.cat(kernel_params, 0)
            states = torch.cat(states, 0)
            mask_params = kernel_params[:, :self.num_mask_params]
            reg_params = kernel_params[:, self.num_mask_params:]
            masks = self.mask_head(mask_branch, mask_params, num_ins)
            regs = self.reg_head(reg_branch, reg_params, num_ins)
            feat_range = masks.permute(0, 1, 3,
                                       2).view(sum(num_ins), w_mask, h_mask)
            feat_range = self.mlp(feat_range)

        return hm, regs, masks, feat_range, states

    def forward_test(
            self,
            inputs,
            hm_thr=0.3,
            max_rtimes=6,
            memory=None,
            hack_seeds=None,
    ):

        def parse_pos(seeds, batchsize, num_classes, h, w, device):
            pos_list = [[p['coord'], p['id_class'] - 1] for p in seeds]
            poses = []
            for p in pos_list:
                [c, r], label = p
                pos = label * h * w + r * w + c
                poses.append(pos)
            poses = torch.from_numpy(np.array(
                poses, np.long)).long().to(device).unsqueeze(1)
            return poses

        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]
        f_deep = f_mask
        m_batchsize = f_deep.size()[0]

        z = self.ctnet_head(f_hm)
        h_hm, w_hm = f_hm.size()[2:]
        hm, params = z['hm'], z['params']
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        h_mask, w_mask = f_mask.size()[2:]
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)

        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        self.debug_mask_branch = mask_branch
        self.debug_reg_branch = reg_branch
        params = params.permute(0, 1, 3, 4,
                                2).contiguous().view(-1, self.rnn_in_channels)

        batch_size, num_classes, h, w = hm.size()
        seeds = self.ctdet_decode(hm, thr=hm_thr)
        if hack_seeds is not None:
            seeds = hack_seeds
        pos_tensor = parse_pos(seeds, batch_size, num_classes, h, w, hm.device)
        pos_tensor = pos_tensor.expand(-1, self.rnn_in_channels)

        if pos_tensor.size()[0] == 0:
            return [], hm
        else:
            kernel_params = []
            num_ins_per_seed = []
            rnn_params = params.gather(0, pos_tensor)
            for idx in range(pos_tensor.size()[0]):
                rnn_feat_input = rnn_params[idx:idx + 1, :]
                rnn_feat_input = rnn_feat_input.reshape(1, -1, 1, 1)
                hidden_h = rnn_feat_input
                hidden_c = rnn_feat_input
                rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)

                if self.zero_hidden_state:
                    hidden_state = None
                else:
                    hidden_state = (hidden_h, hidden_c)
                num_ins_count = 0
                for _ in range(max_rtimes):
                    rnn_out, hidden_state = self.rnn_ceil(
                        inputs=rnn_feat_input,
                        hidden_state=hidden_state,
                        seq_len=1)
                    rnn_out = rnn_out.reshape(1, -1, 1, 1)
                    k_param, state = self.final_fc(rnn_out)
                    k_param = k_param.squeeze(-1).squeeze(-1)
                    state = state.squeeze(-1).squeeze(-1)
                    kernel_params.append(k_param)
                    num_ins_count += 1
                    if torch.argmax(state[0]) == 0:
                        break
                    rnn_feat_input = rnn_out
                    rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)
                num_ins_per_seed.append(num_ins_count)

            num_ins = len(kernel_params)
            kernel_params = torch.cat(kernel_params, 0)
            mask_params = kernel_params[:, :self.num_mask_params]
            reg_params = kernel_params[:, self.num_mask_params:]
            masks = self.mask_head(mask_branch, mask_params, [num_ins])

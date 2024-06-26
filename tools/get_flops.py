import argparse

import torch
from mmcv import Config

from mmdet.models import build_detector
from mmdet.utils import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',default=r"E:\lane-detecion\PycharmProjects\conditional-lane-detection-master\configs\condlanenet\culane\culane_medium_test.py", help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 320],
        help='input image size')
    args = parser.parse_args()
    return args


from thop import profile

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    # input = torch.rand((1, 3, 800, 320), device='cuda')
    # flops, params = profile(model, inputs=(input,))
    # print("FLOPs=", str((flops) / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {(flops)}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

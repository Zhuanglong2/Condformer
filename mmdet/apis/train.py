import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner

from mmdet.core import (DistEvalHook, DistOptimizerHook, EvalHook,
                        Fp16OptimizerHook, build_optimizer)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    # parameters = []
    # WEIGHT = ['module.neck.fpn_convs2.0.conv.weight','module.neck.fpn_convs2.0.conv.bias','module.neck.fpn_convs2.1.conv.weight','module.neck.fpn_convs2.1.conv.bias',
    #           'module.bbox_head.query_embed.weight',
    #           'module.bbox_head.transformer.encoder.layers.0.self_attn.in_proj_weight',
    #           'module.bbox_head.transformer.encoder.layers.0.self_attn.in_proj_bias',
    #           'module.bbox_head.transformer.encoder.layers.0.self_attn.out_proj.weight',
    #           'module.bbox_head.transformer.encoder.layers.0.self_attn.out_proj.bias',
    #           'module.bbox_head.transformer.encoder.layers.0.linear1.weight',
    #           'module.bbox_head.transformer.encoder.layers.0.linear1.bias',
    #           'module.bbox_head.transformer.encoder.layers.0.linear2.weight',
    #           'module.bbox_head.transformer.encoder.layers.0.linear2.bias',
    #           'module.bbox_head.transformer.encoder.layers.0.norm1.weight',
    #           'module.bbox_head.transformer.encoder.layers.0.norm1.bias',
    #           'module.bbox_head.transformer.encoder.layers.0.norm2.weight',
    #           'module.bbox_head.transformer.encoder.layers.0.norm2.bias',
    #           'module.bbox_head.transformer.encoder.layers.1.self_attn.in_proj_weight',
    #           'module.bbox_head.transformer.encoder.layers.1.self_attn.in_proj_bias',
    #           'module.bbox_head.transformer.encoder.layers.1.self_attn.out_proj.weight',
    #           'module.bbox_head.transformer.encoder.layers.1.self_attn.out_proj.bias',
    #           'module.bbox_head.transformer.encoder.layers.1.linear1.weight',
    #           'module.bbox_head.transformer.encoder.layers.1.linear1.bias',
    #           'module.bbox_head.transformer.encoder.layers.1.linear2.weight',
    #           'module.bbox_head.transformer.encoder.layers.1.linear2.bias',
    #           'module.bbox_head.transformer.encoder.layers.1.norm1.weight',
    #           'module.bbox_head.transformer.encoder.layers.1.norm1.bias',
    #           'module.bbox_head.transformer.encoder.layers.1.norm2.weight',
    #           'module.bbox_head.transformer.encoder.layers.1.norm2.bias',
    #           'module.bbox_head.transformer.encoder.query_scale.layers.0.weight',
    #           'module.bbox_head.transformer.encoder.query_scale.layers.0.bias',
    #           'module.bbox_head.transformer.encoder.query_scale.layers.1.weight',
    #           'module.bbox_head.transformer.encoder.query_scale.layers.1.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_qcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_qcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_qpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_qpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_kcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_kcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_kpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_kpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_v_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.sa_v_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.self_attn.out_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.self_attn.out_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.norm1.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.norm1.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_qcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_qcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_qpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_qpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_kcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_kcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_kpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_kpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_v_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_v_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_qpos_sine_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.ca_qpos_sine_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.cross_attn.out_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.cross_attn.out_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.linear1.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.linear1.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.linear2.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.linear2.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.norm2.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.norm2.bias',
    #           'module.bbox_head.transformer.decoder.layers.0.norm3.weight',
    #           'module.bbox_head.transformer.decoder.layers.0.norm3.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_qcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_qcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_qpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_qpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_kcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_kcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_kpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_kpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_v_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.sa_v_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.self_attn.out_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.self_attn.out_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.norm1.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.norm1.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_qcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_qcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_qpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_qpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_kcontent_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_kcontent_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_kpos_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_kpos_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_v_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_v_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_qpos_sine_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.ca_qpos_sine_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.cross_attn.out_proj.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.cross_attn.out_proj.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.linear1.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.linear1.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.linear2.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.linear2.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.norm2.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.norm2.bias',
    #           'module.bbox_head.transformer.decoder.layers.1.norm3.weight',
    #           'module.bbox_head.transformer.decoder.layers.1.norm3.bias',
    #           'module.bbox_head.transformer.decoder.norm.weight',
    #           'module.bbox_head.transformer.decoder.norm.bias',
    #           'module.bbox_head.transformer.decoder.ref_point_head.layers.0.weight',
    #           'module.bbox_head.transformer.decoder.ref_point_head.layers.0.bias',
    #           'module.bbox_head.transformer.decoder.ref_point_head.layers.1.weight',
    #           'module.bbox_head.transformer.decoder.ref_point_head.layers.1.bias',
    #           'module.bbox_head.transformer.decoder.query_scale.layers.0.weight',
    #           'module.bbox_head.transformer.decoder.query_scale.layers.0.bias',
    #           'module.bbox_head.transformer.decoder.query_scale.layers.1.weight',
    #           'module.bbox_head.transformer.decoder.query_scale.layers.1.bias',
    #           'module.bbox_head.coodinate_embed.layers.0.weight',
    #           'module.bbox_head.coodinate_embed.layers.0.bias',
    #           'module.bbox_head.coodinate_embed.layers.1.weight',
    #           'module.bbox_head.coodinate_embed.layers.1.bias',
    #           'module.bbox_head.coodinate_embed.layers.2.weight',
    #           'module.bbox_head.coodinate_embed.layers.2.bias',
    #           'module.bbox_head.class_embed.weight',
    #           'module.bbox_head.class_embed.bias'
    #           ]
    # for name, p in model.named_parameters():
    #     # print(name)
    #     if name in WEIGHT:
    #         parameters.append(p)
    # optimizer = torch.optim.Adam(
    #     parameters, lr=cfg.optimizer['lr'],betas=cfg.optimizer['betas'], eps= cfg.optimizer['eps'])

    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

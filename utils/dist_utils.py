# some code in this file is adapted from
# https://github.com/facebookresearch/moco
# Original Copyright 2020 Facebook, Inc. and its affiliates. Licensed under the CC-BY-NC 4.0 License.
# Modifications Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import sys
import random
import datetime
import torch
import torch.distributed as dist


@torch.no_grad()
def batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def init_distributed_mode(args):
    if is_dist_avail_and_initialized():
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(random.randint(0, 9999) + 40000)
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    print("Use GPU: {} ranked {} out of {} gpus for training".format(args.gpu, args.rank, args.world_size))
    if args.multiprocessing_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            timeout=datetime.timedelta(hours=5),
            rank=args.rank,
        )
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        dist.barrier()

    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def all_reduce_mean(x):
    # reduce tensore for DDP
    # source: https://raw.githubusercontent.com/NVIDIA/apex/master/examples/imagenet/main_amp.py
    world_size = get_world_size()
    if world_size > 1:
        rt = x.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= world_size
        return rt
    else:
        return x

# def dist_init(port=23456):
#
#     def init_parrots(host_addr, rank, local_rank, world_size, port):
#         os.environ['MASTER_ADDR'] = str(host_addr)
#         os.environ['MASTER_PORT'] = str(port)
#         os.environ['WORLD_SIZE'] = str(world_size)
#         os.environ['RANK'] = str(rank)
#         torch.distributed.init_process_group(backend="nccl")
#         torch.cuda.set_device(local_rank)
#
#     def init(host_addr, rank, local_rank, world_size, port):
#         host_addr_full = 'tcp://' + host_addr + ':' + str(port)
#         torch.distributed.init_process_group("nccl", init_method=host_addr_full,
#                                             rank=rank, world_size=world_size)
#         torch.cuda.set_device(local_rank)
#         assert torch.distributed.is_initialized()
#
#
#     def parse_host_addr(s):
#         if '[' in s:
#             left_bracket = s.index('[')
#             right_bracket = s.index(']')
#             prefix = s[:left_bracket]
#             first_number = s[left_bracket+1:right_bracket].split(',')[0].split('-')[0]
#             return prefix + first_number
#         else:
#             return s
#
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         local_rank = int(os.environ['LOCAL_RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         ip = 'env://'
#
#     elif 'SLURM_PROCID' in os.environ:
#         rank = int(os.environ['SLURM_PROCID'])
#         local_rank = int(os.environ['SLURM_LOCALID'])
#         world_size = int(os.environ['SLURM_NTASKS'])
#         ip = parse_host_addr(os.environ['SLURM_STEP_NODELIST'])
#     else:
#         raise RuntimeError()
#
#     if torch.__version__ == 'parrots':
#         init_parrots(ip, rank, local_rank, world_size, port)
#     else:
#         init(ip, rank, local_rank, world_size, port)
#
#     return rank, local_rank, world_size


# https://github.com/facebookresearch/msn
class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" multi-label classification
https://github.com/d-li14/face-attribute-prediction
https://kkaryl.github.io/project/celeba/
https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
"""

__author__ = "GZ"

import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from classy_vision.generic.distributed_util import is_distributed_training_run

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from data.celeba import CelebA
import data.transforms as data_transforms
from data.transforms import AddGaussianNoise
import backbone as customized_models
from models import farl
from engine import validate_multilabel
from utils import utils, get_norm
from utils.dist_utils import init_distributed_mode, all_reduce_mean

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='in1k',
                    help='name of dataset')
parser.add_argument('--data-root', default="",
                    help='root of dataset folder')
parser.add_argument('--trainindex', default=None, type=str, metavar='PATH',
                    help='path to train annotation (default: None)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default="sgd", type=str, help='[sgd, adamw]')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument("--lr_head", default=0.02, type=float, help="initial learning rate - head")
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--scheduler', default="cos", type=str, help='[cos, step, exp]')
parser.add_argument('--decay_epochs', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument("--gamma", type=float, default=0.1, help="lr decay factor")

parser.add_argument('--save-dir', default="ckpts",
                    help='checkpoint directory')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# dist
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; """)
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to self-supervised pretrained checkpoint')
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
parser.add_argument('--amp', action='store_true', help='use automatic mixed precision training')

parser.add_argument('--model_type', default="ours", type=str, help='type of model')
parser.add_argument('--image_size', type=int, default=224, help='image size')
parser.add_argument('--data_norm', default="vggface2", type=str, help='type of data/transformation norm')
parser.add_argument('--finetune', action='store_true', help='fine-tune downstream backbone')

NUM_CLASSES = {'celeba': 40}
best_acc1 = 0


# https://github.com/d-li14/face-attribute-prediction/blob/master/models/resnet.py
class fc_block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.2):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x


class AttNeck(nn.Module):
    def __init__(self, num_attributes=40, drop_rate=0.2):
        super().__init__()

        self.stem = fc_block(2048, 512, drop_rate)
        self.classifier = nn.Sequential(fc_block(512, 256, drop_rate), nn.Linear(256, num_attributes))

        # self.classifier = nn.Sequential(fc_block(2048, 512, drop_rate), nn.Linear(512, num_attributes))

        # self.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(2048, num_attributes))

        # self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        y = self.classifier(x)

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def main(args):
	global best_acc1
	# args.gpu = args.local_rank

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	# create model
	print("=> creating model '{}'".format(args.arch))
	if args.model_type == "ours":
		norm = get_norm(args.norm)
		model = models.__dict__[args.arch](num_classes=NUM_CLASSES[args.dataset], norm_layer=norm)
	elif args.model_type == "farl":
		model_path = "./checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
		vt = farl.FaRLVisualFeatures(model_type="base", model_path=model_path, forced_input_resolution=224)
		model = farl.FaRLClsWrapper(vt, num_classes=NUM_CLASSES[args.dataset])

	if not args.finetune:
		# freeze all layers but the last fc
		for name, param in model.named_parameters():
			if name not in ['fc.weight', 'fc.bias']:
				param.requires_grad = False
	# init the fc layer
	model.fc.weight.data.normal_(mean=0.0, std=0.01)
	model.fc.bias.data.zero_()

	model.fc = AttNeck(NUM_CLASSES[args.dataset], drop_rate=0.2)
	print(model)

	# load from pre-trained, before DistributedDataParallel constructor
	if args.pretrained:
		if os.path.isfile(args.pretrained):
			print("=> loading checkpoint '{}'".format(args.pretrained))
			checkpoint = torch.load(args.pretrained, map_location="cpu")

			# rename moco pre-trained keys
			state_dict = checkpoint['state_dict']
			model_prefix = 'module.' + args.model_prefix
			# print(state_dict.keys())
			for k in list(state_dict.keys()):
				# retain only student model up to before the embedding layer
				if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
					# remove prefix
					new_key = k.replace(model_prefix + '.', "")
					state_dict[new_key] = state_dict[k]
				# delete renamed or unused k
				del state_dict[k]
			print(state_dict.keys())
			args.start_epoch = 0
			msg = model.load_state_dict(state_dict, strict=False)
			# assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
			if len(msg.missing_keys) > 0:
				print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
			if len(msg.unexpected_keys) > 0:
				print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
			print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))

		# elif os.path.isfile(args.pretrained):
		# 	print("=> loading checkpoint '{}'".format(args.pretrained))
		# 	state_dict = torch.load(args.pretrained, map_location="cpu")
		# 	print(state_dict.keys())
		# 	args.start_epoch = 0
		# 	msg = model.load_state_dict(state_dict, strict=False)
		# 	assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
		# 	if len(msg.missing_keys) > 0:
		# 		print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
		# 	if len(msg.unexpected_keys) > 0:
		# 		print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
		# 	print("=> loaded pre-trained model '{}'".format(args.pretrained))

		else:
			print("=> no checkpoint found at '{}'".format(args.pretrained))

	if args.multiprocessing_distributed:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model.cuda()
		args.batch_size = int(args.batch_size / args.world_size)
		args.workers = int((args.workers + args.world_size - 1) / args.world_size)
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
	else:
		model.cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

	# set optimizer
	if not args.finetune:
		# optimize only the linear classifier
		params = list(filter(lambda p: p.requires_grad, model.parameters()))
		assert len(params) == 2  # fc.weight, fc.bias
	else:
		trunk_parameters = []
		head_parameters, head_names = [], []
		for name, param in model.named_parameters():
			if name.startswith('fc') or name.startswith('module.fc'):
				head_parameters.append(param)
				head_names.append(name)
			else:
				trunk_parameters.append(param)
		# assert len(head_parameters) == 2
		print("classifier params: {} using lr {}".format(head_names, args.lr_head))
		print("the rest params using lr {}".format(args.lr))
		params = [{'params': trunk_parameters},
		          {'params': head_parameters, 'lr': args.lr_head}]

	if args.optimizer == "sgd":
		optimizer = torch.optim.SGD(
			params,
			lr=args.lr,
			momentum=args.momentum,
			weight_decay=args.weight_decay,
		)
	elif args.optimizer == 'adamw':
		optimizer = torch.optim.AdamW(
			params,
			lr=args.lr,
			weight_decay=args.weight_decay,
		)

	scaler = torch.cuda.amp.GradScaler() if args.amp else None

	# set scheduler
	# if args.scheduler == "cos":
	# 	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	# elif args.scheduler == "step":
	# 	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=args.gamma)
	# elif args.scheduler == "exp":
	# 	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
			      .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# Data loading code
	normalize = transforms.Normalize(mean=data_transforms.IMG_MEAN[args.data_norm],
	                                 std=data_transforms.IMG_STD[args.data_norm])
	transform_train = transforms.Compose([
		# transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.2)),
		transforms.Resize([args.image_size, args.image_size]),
		transforms.RandomHorizontalFlip(),
		transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
		transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
		transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
		transforms.ToTensor(),
		# transforms.RandomApply([AddGaussianNoise(0., 0.02)], p=0.5),
		normalize,
		# transforms.RandomErasing()
	])
	transform_test = transforms.Compose([
		transforms.Resize([args.image_size, args.image_size]),
		# transforms.CenterCrop(args.image_size),
		transforms.ToTensor(),
		normalize,
	])

	if args.dataset.lower() == "celeba":
		train_dataset = CelebA(root=args.data_root, split="train", transform=transform_train, crop=False)
		val_dataset = CelebA(root=args.data_root, split="test", transform=transform_test, crop=False)

	print(train_dataset)
	print(transform_train)

	train_sampler = None
	if args.multiprocessing_distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True, persistent_workers=True)

	if args.evaluate:
		validate_multilabel(val_loader, model, criterion, args)
		return

	best_epoch = args.start_epoch
	for epoch in range(args.start_epoch, args.epochs):
		if args.multiprocessing_distributed:
			train_sampler.set_epoch(epoch)
		# adjust_learning_rate(optimizer, epoch, args)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, scaler, epoch, args)

		# scheduler.step()

		if (epoch + 1) % args.eval_freq == 0:
			# evaluate on validation set
			acc1 = validate_multilabel(val_loader, model, criterion, args)

			# remember best acc@1 and save checkpoint
			is_best = acc1 > best_acc1
			best_acc1 = max(acc1, best_acc1)
			if is_best:
				best_epoch = epoch

			# if not args.multiprocessing_distributed or (args.multiprocessing_distributed
			#                                             and args.local_rank % args.world_size == 0):
			# 	save_checkpoint({
			# 		'epoch': epoch + 1,
			# 		'arch': args.arch,
			# 		'state_dict': model.state_dict(),
			# 		'best_acc1': best_acc1,
			# 		'optimizer': optimizer.state_dict(),
			# 	}, is_best,
			# 		dir=os.path.join(args.save_dir, "checkpoint_fer"),
			# 		filename='{:04d}.pth.tar'.format(epoch)
			# 	)
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
		                                            and args.rank % args.world_size == 0):
			if epoch == args.start_epoch and not args.finetune:
				sanity_check(model.state_dict(), args.pretrained, args)

	print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))


def train(train_loader, model, criterion, optimizer, scaler, epoch, args):
	batch_time = utils.AverageMeter('Time', ':6.3f')
	data_time = utils.AverageMeter('Data', ':6.3f')
	losses = utils.AverageMeter('Loss', ':.4e')
	top1 = utils.AverageMeter('Acc@1', ':6.2f')
	top5 = utils.AverageMeter('Acc@5', ':6.2f')
	lr_trunk = optimizer.param_groups[0]['lr']
	if args.finetune:
		lr_head = optimizer.param_groups[1]["lr"]
	else:
		lr_head = lr_trunk
	progress = utils.ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}/{}]\t"
		       "LR trunk/head: {:.7f}/{:.7f}\t".format(epoch, args.epochs, lr_trunk, lr_head))

	"""
	Switch to eval mode:
	Under the protocol of linear classification on frozen features/models,
	it is not legitimate to change any part of the pre-trained model.
	BatchNorm in train mode may revise running mean/std (even if it receives
	no gradient), which are part of the model parameters too.
	"""
	if args.finetune:
		model.train()
	else:
		model.eval()
		model.fc.train()

	end = time.time()
	for i, (images, target, _) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
		target = target.cuda(args.gpu, non_blocking=True).float()

		if scaler is None:
			# compute output
			output = model(images)
			loss = criterion(output, target)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		else:
			optimizer.zero_grad()
			with torch.cuda.amp.autocast():
				output = model(images)
				loss = criterion(output, target)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

		# measure accuracy and record loss
		acc1 = utils.accuracy_multilabel(torch.sigmoid(output), target)

		if is_distributed_training_run():
			# torch.distributed.barrier()
			acc1 = all_reduce_mean(acc1)

		losses.update(loss.item(), images.size(0))
		top1.update(acc1.item(), images.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.display(i)


def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
	os.makedirs(dir, exist_ok=True)
	file_path = os.path.join(dir, filename)
	torch.save(state, file_path)


def sanity_check(state_dict, pretrained_weights, args):
	"""
	Linear classifier should not change any weights other than the linear layer.
	This sanity check asserts nothing wrong happens (e.g., BN stats updated).
	"""
	print("=> loading '{}' for sanity check".format(pretrained_weights))
	checkpoint = torch.load(pretrained_weights, map_location="cpu")
	state_dict_pre = checkpoint['state_dict']

	model_prefix = 'module.' + args.model_prefix + '.'
	for k in list(state_dict.keys()):
		# only ignore fc layer
		if 'fc.weight' in k or 'fc.bias' in k:
			continue

		# name in pretrained model
		k_pre = model_prefix + k[len('module.'):] \
			if k.startswith('module.') else model_prefix + k
		# for BYOL model
		k_pre = k_pre.replace('backbone.fc.', 'neck.mlp.')

		assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
			'{} is changed in linear classifier training.'.format(k)

	print("=> sanity check passed.")


# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     if args.cos:  # cosine lr schedule
#         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
#     else:  # stepwise lr schedule
#         for milestone in args.schedule:
#             lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
	"""Decay the learning rate with half-cycle cosine after warmup"""
	num_groups = len(optimizer.param_groups)
	assert 1 <= num_groups <= 2
	lrs = []
	if num_groups == 1:
		lrs += [args.lr]
	elif num_groups == 2:
		lrs += [args.lr, args.lr_head]
	assert len(lrs) == num_groups
	for group_id, param_group in enumerate(optimizer.param_groups):
		lr = lrs[group_id]
		if args.scheduler == "cos":
			if epoch < args.warmup_epochs:
				lr = lr * epoch / args.warmup_epochs
			else:
				lr = args.min_lr + (lr - args.min_lr) * 0.5 * \
				     (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
		elif args.scheduler == "step":
			for milestone in args.decay_epochs:
				lr *= 0.1 if epoch >= milestone else 1.
		param_group['lr'] = lr


if __name__ == '__main__':
	opt = parser.parse_args()

	# _, opt.local_rank, opt.world_size = dist_init(opt.port)
	# cudnn.benchmark = True
	#
	# # suppress printing if not master
	# if dist.get_rank() != 0:
	#     def print_pass(*args, **kwargs):
	#         pass
	#     builtins.print = print_pass

	init_distributed_mode(opt)
	print(opt)

	main(opt)

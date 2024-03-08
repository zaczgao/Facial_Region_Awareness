# Original copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import os
import numpy as np
import shutil
from sklearn.metrics import accuracy_score
import skimage.io

import torch


def load_netowrk(model, path, checkpoint_key="net"):
	if os.path.isfile(path):
		print("=> loading checkpoint '{}'".format(path))
		checkpoint = torch.load(path, map_location="cpu")

		# rename pre-trained keys
		state_dict = checkpoint[checkpoint_key]
		state_dict_new = {k.replace("module.", ""): v for k, v in state_dict.items()}

		msg = model.load_state_dict(state_dict_new)
		assert set(msg.missing_keys) == set()

		print("=> loaded pre-trained model '{}'".format(path))
	else:
		print("=> no checkpoint found at '{}'".format(path))


def save_checkpoint(state, is_best, epoch, args, filename='checkpoint.pth.tar'):
    filename = os.path.join(args.save_dir, filename)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))
    if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
        shutil.copyfile(filename,  os.path.join(args.save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))
    if not args.cos:
        if (epoch + 1) in args.schedule:
            shutil.copyfile(filename,  os.path.join(args.save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_multilabel(output, target, threshold=0.5):
	"""
	https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation
	"""
	with torch.no_grad():
		batch_size, n_class = target.shape
		pred = (output >= threshold).to(torch.float32)

		acc = (pred == target).float().sum() * 100.0 / (batch_size * n_class)

	# acc =  sklearn.metrics.accuracy_score(gt_S,pred_S)
	# f1m = sklearn.metrics.f1_score(gt_S,pred_S,average = 'macro', zero_division=1)
	# f1mi = sklearn.metrics.f1_score(gt_S,pred_S,average = 'micro', zero_division=1)
	# print('f1_Macro_Score{}'.format(f1m))
	# print('f1_Micro_Score{}'.format(f1mi))
	# print('Accuracy{}'.format(acc))

	return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class InstantMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def denormalize_batch(batch, mean, std):
	"""denormalize for visualization"""
	dtype = batch.dtype
	mean = torch.as_tensor(mean, dtype=dtype, device=batch.device)
	std = torch.as_tensor(std, dtype=dtype, device=batch.device)
	mean = mean.view(-1, 1, 1)
	std = std.view(-1, 1, 1)
	batch = batch * std + mean
	return batch

def dump_image(imgNorm, mean, std, filepath=None, verbose=False):
	"""Denormalizes the output image and optionally plots the landmark coordinates onto the image

	Args:
		normalized_image (torch.tensor): Image reconstruction output from the model (normalized)
		landmark_coords (torch.tensor): x, y coordinates in normalized range -1 to 1
		out_name (str, optional): file to write to
	Returns:
		np.array: uint8 image data stored in numpy format
	"""
	if imgNorm.dim() < 4:
		imgNorm = imgNorm.unsqueeze(0)

	img = denormalize_batch(imgNorm, mean, std)
	img = np.clip(img.cpu().numpy(), 0, 1)
	img = (img.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

	if filepath is not None:
		skimage.io.imsave(filepath, img[0])

	if verbose:
		num = min(img.shape[0], 9)
		show_images(img[:num], 3, 3)
		plt.show()
	return img


def calc_params(net, verbose=False):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	if verbose:
		print(net)
	print('Total number of parameters : %.3f M' % (num_params / 1e6))

	return num_params


if __name__ == '__main__':
	output = torch.tensor([[0.35,0.4,0.9], [0.2,0.6,0.8]])
	target = torch.tensor([[1, 0, 1], [0, 1, 1]])
	acc = accuracy_multilabel(output, target)
	print(acc)

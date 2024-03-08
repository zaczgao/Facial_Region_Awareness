# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import random
from io import BytesIO
from PIL import Image
from PIL import ImageOps, ImageFilter
import torch
from torchvision import transforms

from .randaugment import RandAugment


# RGB mean tensor([0.5885, 0.4407, 0.3724])
# RGB std tensor([0.2271, 0.1961, 0.1827])

# RGB mean tensor([0.5231, 0.4044, 0.3489])
# RGB std tensor([0.2536, 0.2194, 0.2070])

IMG_MEAN = {"vggface2": [0.5231, 0.4044, 0.3489],
            "laionface": [0.48145466, 0.4578275, 0.40821073],
            "in1k": [0.485, 0.456, 0.406],
            "in100": [0.485, 0.456, 0.406]}
IMG_STD = {"vggface2": [0.2536, 0.2194, 0.2070],
           "laionface": [0.26862954, 0.26130258, 0.27577711],
           "in1k": [0.229, 0.224, 0.225],
           "in100": [0.229, 0.224, 0.225]}


class Solarize(object):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        tensor = torch.clamp(tensor, min=0., max=1.)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)


class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high

    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = transforms.functional.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = transforms.functional.resize(im, (H, W))
        return im


def get_augmentations(aug_type, dataset):
    normalize = transforms.Normalize(mean=IMG_MEAN[dataset.lower()],
                                     std=IMG_STD[dataset.lower()])

    default_train_augs = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    default_val_augs = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    appendix_augs = [
        transforms.ToTensor(),
        normalize,
    ]
    if aug_type == 'DefaultTrain':
        augs = default_train_augs + appendix_augs
    elif aug_type == 'DefaultVal':
        augs = default_val_augs + appendix_augs
    elif aug_type == 'RandAugment':
        augs = default_train_augs + [RandAugment(n=2, m=10)] + appendix_augs
    elif aug_type == 'MoCoV1':
        augs = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip()
        ] + appendix_augs
    elif aug_type == 'MoCoV2':
        augs = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ] + appendix_augs
    else:
        raise NotImplementedError('augmentation type not found: {}'.format(aug_type))

    return augs


def get_transforms(aug_type, dataset="in1k"):
    augs = get_augmentations(aug_type, dataset)
    return transforms.Compose(augs)


def get_byol_tranforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
        transforms.RandomApply([Solarize()], p=0.),
        transforms.ToTensor(),
        normalize
    ]
    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    transform1 = transforms.Compose(augmentation1)
    transform2 = transforms.Compose(augmentation2)
    return transform1, transform2


def get_vggface_tranforms(image_size=128):
    normalize = transforms.Normalize(mean=IMG_MEAN["vggface2"],
                                     std=IMG_STD["vggface2"])

    augmentation1 = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        # transforms.Resize([image_size, image_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
        transforms.RandomApply([Solarize()], p=0.),
        transforms.ToTensor(),
        normalize
    ]
    augmentation2 = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
        # transforms.Resize([image_size, image_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    transform1 = transforms.Compose(augmentation1)
    transform2 = transforms.Compose(augmentation2)
    return transform1, transform2


class TwoCropsTransform:
    """Take two random crops of one image."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        out1 = self.transform1(x)
        out2 = self.transform2(x)
        return out1, out2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        names = ['transform1', 'transform2']
        for idx, t in enumerate([self.transform1, self.transform2]):
            format_string += '\n'
            t_string = '{0}={1}'.format(names[idx], t)
            t_string_split = t_string.split('\n')
            t_string_split = ['    ' + tstr for tstr in t_string_split]
            t_string = '\n'.join(t_string_split)
            format_string += '{0}'.format(t_string)
        format_string += '\n)'
        return format_string


if __name__ == '__main__':
    from utils.utils import dump_image

    # Ryan_Gosling Emily_VanCamp
    img = Image.open("./vis_data/0008_01.jpg")

    augment = get_vggface_tranforms(image_size=224)
    img1 = augment[0](img)
    img2 = augment[1](img)

    save_dir = "./output"
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, "{}.png".format("img1"))
    dump_image(img1, IMG_MEAN["vggface2"], IMG_STD["vggface2"], filepath=filepath)

    filepath = os.path.join(save_dir, "{}.png".format("img2"))
    dump_image(img2, IMG_MEAN["vggface2"], IMG_STD["vggface2"], filepath=filepath)

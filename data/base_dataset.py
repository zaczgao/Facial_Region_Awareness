# Copyright Lang Huang (laynehuang@outlook.com). All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image

try:
    import mc
except ImportError:
    mc = None
import io


# class DatasetCache(data.Dataset):
#     def __init__(self):
#         super().__init__()
#         self.initialized = False
#
#
#     def _init_memcached(self):
#         if not self.initialized:
#             server_list_config_file = "/mnt/cache/share/memcached_client/server_list.conf"
#             client_config_file = "/mnt/cache/share/memcached_client/client.conf"
#             self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
#             self.initialized = True
#
#     def load_image(self, filename):
#         self._init_memcached()
#         value = mc.pyvector()
#         self.mclient.Get(filename, value)
#         value_str = mc.ConvertBuffer(value)
#
#         buff = io.BytesIO(value_str)
#         with Image.open(buff) as img:
#             img = img.convert('RGB')
#         return img
#
#
#
# class BaseDataset(DatasetCache):
#     def __init__(self, mode='train', max_class=1000, aug=None,
#                         prefix='/mnt/cache/share/images/meta',
#                         image_folder_prefix='/mnt/cache/share/images/'):
#         super().__init__()
#         self.initialized = False
#
#         if mode == 'train':
#             image_list = os.path.join(prefix, 'train.txt')
#             self.image_folder = os.path.join(image_folder_prefix, 'train')
#         elif mode == 'test':
#             image_list = os.path.join(prefix, 'test.txt')
#             self.image_folder = os.path.join(image_folder_prefix, 'test')
#         elif mode == 'val':
#             image_list = os.path.join(prefix, 'val.txt')
#             self.image_folder = os.path.join(image_folder_prefix, 'val')
#         else:
#             raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, val]')
#
#
#         self.samples = []
#         with open(image_list) as f:
#             for line in f:
#                 name, label = line.split()
#                 label = int(label)
#                 if label < max_class:
#                     self.samples.append((label, name))
#
#         if aug is None:
#             if mode == 'train':
#                 self.transform = transforms.Compose([
#                     transforms.RandomResizedCrop(224),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                             std=[0.229, 0.224, 0.225])
#                 ])
#             else:
#                 self.transform = transforms.Compose([
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                 std=[0.229, 0.224, 0.225]),
#                 ])
#
#         else:
#             self.transform = aug
#
#
# def get_keep_index(samples, percent, num_classes, shuffle=False):
#     labels = np.array([sample[0] for sample in samples])
#     keep_indexs = []
#     for i in range(num_classes):
#         idx = np.where(labels == i)[0]
#         num_sample = len(idx)
#         label_per_class = min(max(1, round(percent * num_sample)), num_sample)
#         if shuffle:
#             np.random.shuffle(idx)
#         keep_indexs.extend(idx[:label_per_class])
#
#     return keep_indexs
#
#
# class ImageNet(BaseDataset):
#     def __init__(self, mode='train', max_class=1000, num_classes=1000, transform=None,
#                        percent=1., shuffle=False, **kwargs):
#         super().__init__(mode, max_class, aug=transform, **kwargs)
#
#         assert 0 <= percent <= 1
#         if percent < 1:
#             keep_indexs = get_keep_index(self.samples, percent, num_classes, shuffle)
#             self.samples = [self.samples[i] for i in keep_indexs]
#
#     def __len__(self):
#         return self.samples.__len__()
#
#     def __getitem__(self, index):
#         label, name = self.samples[index]
#         filename = os.path.join(self.image_folder, name)
#         img = self.load_image(filename)
#         return self.transform(img), label, index
#
#
# class ImageNetWithIdx(BaseDataset):
#     def __init__(self, mode='train', max_class=1000, num_classes=1000, transform=None,
#                        idx=None, shuffle=False, **kwargs):
#         super().__init__(mode, max_class, aug=transform, **kwargs)
#
#         assert idx is not None
#         with open(idx, "r") as fin:
#             samples = [line.strip().split(" ") for line in fin.readlines()]
#         self.samples = samples
#         print(f"Len of training set: {len(self.samples)}")
#
#     def __len__(self):
#         return self.samples.__len__()
#
#     def __getitem__(self, index):
#         label, name = self.samples[index]
#         filename = os.path.join(self.image_folder, name)
#         img = self.load_image(filename)
#         return self.transform(img), int(label), index
#
#
# class ImageNet100(ImageNet):
#     def __init__(self, **kwargs):
#         super().__init__(
#             num_classes=100,
#             prefix='/mnt/lustre/huanglang/research/selfsup/data/imagenet-100/',
#             image_folder_prefix='/mnt/lustre/huanglang/research/selfsup/data/images',
#             **kwargs)
#
# class ImageFolderWithPercent(ImageFolder):
#
#     def __init__(self, root, transform=None, target_transform=None,
#                  loader=default_loader, is_valid_file=None, percent=1.0, shuffle=False):
#         super().__init__(root, transform=transform, target_transform=target_transform,
#                          loader=loader, is_valid_file=is_valid_file)
#         assert 0 <= percent <= 1
#         if percent < 1:
#             keep_indexs = get_keep_index(self.targets, percent, len(self.classes), shuffle)
#             self.samples = [self.samples[i] for i in keep_indexs]
#             self.targets = [self.targets[i] for i in keep_indexs]
#             self.imgs = self.samples
#
#
# class ImageFolderWithIndex(ImageFolder):
#
#     def __init__(self, root, indexs=None, transform=None, target_transform=None,
#                  loader=default_loader, is_valid_file=None):
#         super().__init__(root, transform=transform, target_transform=target_transform,
#                          loader=loader, is_valid_file=is_valid_file)
#         if indexs is not None:
#             self.samples = [self.samples[i] for i in indexs]
#             self.targets = [self.targets[i] for i in indexs]
#             self.imgs = self.samples


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class ImageFolderSubset(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, class_path, root, transform, **kwargs):
        super().__init__(root, transform, **kwargs)
        self.class_path = class_path
        new_samples, sorted_classes = self.get_class_samples()
        self.imgs = self.samples = new_samples  # len=126689
        self.classes = sorted_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
        self.targets = [s[1] for s in self.samples]

    def get_class_samples(self):
        classes = open(self.class_path).readlines()
        classes = [m.strip() for m in classes]
        classes = set(classes)
        class_to_sample = [[os.path.basename(os.path.dirname(m[0])), m] for m in self.imgs]
        selected_samples = [m[1] for m in class_to_sample if m[0] in classes]

        sorted_classes = sorted(list(classes))
        target_mapping = {self.class_to_idx[k]: j for j, k in enumerate(sorted_classes)}

        valid_pairs = [[m[0], target_mapping[m[1]]] for m in selected_samples]
        return valid_pairs, sorted_classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def get_dataset(dataset, mode, transform, data_root=None, **kwargs):
    data_dir = os.path.join(data_root, mode)
    if mode == "val" and "ImageNet" in data_root and "nobackup_mmv_ioannisp" in data_root:
        data_dir = "/import/nobackup_mmv_ioannisp/zg002/data/ImageNet/val"
    in100_class_path = "./data/imagenet100.txt"

    if dataset.lower() == 'in1k':
        return ImageFolderInstance(data_dir, transform=transform)
    elif dataset.lower() == 'in100':
        return ImageFolderSubset(in100_class_path, data_dir, transform)
    elif dataset.lower() == "vggface2":
        return ImageFolderInstance(data_dir, transform=transform)
    # elif dataset == 'in1k_idx':
    #     return ImageNetWithIdx(mode, transform=transform, **kwargs)
    # else:   # ImageFolder
    #     data_dir = os.path.join(data_root, mode)
    #     assert os.path.isdir(data_dir)
    #     return ImageFolderWithPercent(data_dir, transform, **kwargs)


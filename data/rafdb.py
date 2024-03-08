#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class RAFDB(data.Dataset):
    def __init__(self, root='/media/jiaren/DataSet/basic/', split='train', transform=None):
        super().__init__()

        self.root = root

        image_list_file = os.path.join(root, "EmoLabel", "list_patition_label.txt")
        self.image_list_file = image_list_file
        self.split = split
        self.transform = transform

        self.samples = []
        self.targets = []
        with open(self.image_list_file, 'r') as f:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()
                img_file = img_file.split(' ')
                if split in img_file[0]:
                    self.samples.append(os.path.join(root, "Image", "aligned", img_file[0][:-4]+'_aligned.jpg'))
                    self.targets.append(int(img_file[1]) - 1)
                    
    def __getitem__(self, index):
        img_file = self.samples[index]
        image = Image.open(img_file)

        if image.mode != 'RGB':
            image = image.convert("RGB")

        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target, index

    def __len__(self):
        return len(self.samples) #12271 #


if __name__ == '__main__':
    display_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    split = "train"
    dataset = RAFDB(root="../data/RAFDB/basic", split=split, transform=display_transform)
    print(len(dataset))
    print(set(dataset.targets))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                         drop_last=False)

    with torch.no_grad():
        for i, (images, target, _) in enumerate(tqdm(loader)):
            img = np.clip(images.cpu().numpy(), 0, 1)  # [0, 1]
            img = img.transpose(0, 2, 3, 1)
            img = (img * 255).astype(np.uint8)
            img = img.squeeze()

            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            axs.imshow(img)
            axs.axis("off")
            plt.show()

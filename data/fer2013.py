#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

__author__ = "GZ"

import os
import sys
import csv
import pathlib
import numpy as np
from tqdm import tqdm
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


# List of folders for training, validation and test.
folder_names = {'Training'   : 'FER2013Train',
                'PublicTest' : 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}


class FER2013(VisionDataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        convert_rgb=False
    ) -> None:
        self._split = split
        assert split in ['Training', 'PublicTest', 'PrivateTest']
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.convert_rgb = convert_rgb

        base_folder = pathlib.Path(self.root)
        data_file = base_folder / "fer2013.csv"

        self._samples = []
        with open(data_file, "r", newline="") as file:
            for row in csv.DictReader(file):
                if split == row["Usage"]:
                    data = (
                        torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                        int(row["emotion"]) if "emotion" in row else None,
                    )
                    self._samples.append(data)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.convert_rgb:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, idx


    def extra_repr(self) -> str:
        return f"split={self._split}"


class FERplus(VisionDataset):
    """
    https://github.com/microsoft/FERPlus/blob/master/src/ferplus.py
    https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
    """
    def __init__(
        self,
        root: str,
        split: str = "Training",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        convert_rgb=False
    ) -> None:
        self._split = split
        assert split in ['Training', 'PublicTest', 'PrivateTest']
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.convert_rgb = convert_rgb
        self.per_emotion_count = None

        # Default values
        self.emotion_count = 8

        # Load data
        self.loaded_data = self._load()
        print('Size of the loaded set: {}'.format(self.loaded_data[0].shape[0]))

    def __len__(self):
        return self.loaded_data[0].shape[0]

    def __getitem__(self, idx):
        image = self.loaded_data[0][idx]
        image = Image.fromarray(image)
        target = self.loaded_data[1][idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target, idx

    # @staticmethod
    # def get_class(idx):
    #     classes = {
    #         0: 'Neutral',
    #         1: 'Happy',
    #         2: 'Sad',
    #         3: 'Surprise',
    #         4: 'Fear',
    #         5: 'Disgust',
    #         6: 'Anger',
    #         7: 'Contempt'}
    #
    #     return classes[idx]
    #
    # @staticmethod
    # def _parse_to_label(idx):
    #     """
    #     Parse labels to make them compatible with AffectNet.
    #     :param idx:
    #     :return:
    #     """
    #     emo_to_return = np.argmax(idx)
    #
    #     if emo_to_return == 2:
    #         emo_to_return = 3
    #     elif emo_to_return == 3:
    #         emo_to_return = 2
    #     elif emo_to_return == 4:
    #         emo_to_return = 6
    #     elif emo_to_return == 6:
    #         emo_to_return = 4
    #
    #     return emo_to_return

    @staticmethod
    def _process_data(emotion_raw):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal)
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        # find the peak value of the emo_raw list
        maxval = max(emotion_raw)
        if maxval > 0.5 * sum_list:
            emotion[np.argmax(emotion_raw)] = maxval
        else:
            emotion = emotion_unknown  # force setting as unknown

        return [float(i) / sum(emotion) for i in emotion]

    def _load(self):
        csv_label = []
        data, labels = [], []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int32)

        path_folders_images = os.path.join(self.root, 'Images', folder_names[self._split])
        path_folders_labels = os.path.join(self.root, 'Labels', folder_names[self._split])

        with open(os.path.join(path_folders_labels, "label.csv")) as csvfile:
            lines = csv.reader(csvfile)
            for row in lines:
                csv_label.append(row)

        for l in csv_label:
            emotion_raw = list(map(float, l[2:len(l)]))
            emotion = self._process_data(emotion_raw)
            idx = np.argmax(emotion)

            if idx < self.emotion_count:  # not unknown or non-face
                self.per_emotion_count[idx] += 1

                # emotion = emotion[:-2]
                # emotion = [float(i) / sum(emotion) for i in emotion]
                # emotion = self._parse_to_label(emotion)

                image = Image.open(os.path.join(path_folders_images, l[0]))
                if self.convert_rgb:
                    image = image.convert("RGB")
                image = np.array(image)

                box = list(map(int, l[1][1:-1].split(',')))
                if box[-1] != 48:
                    print("[INFO] Face is not centralized.")
                    print(os.path.join(path_folders_images, l[0]))
                    print(box)
                    exit(-1)

                image = image[box[0]:box[2], box[1]:box[3], :]

                data.append(image)
                labels.append(idx)

        return [np.array(data), np.array(labels)]


if __name__ == '__main__':
    display_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    split = "PrivateTest"
    # dataset = FER2013(root="../data/FER/fer2013", split=split, transform=display_transform)
    dataset = FERplus(root="../data/FER/FERPlus/data", split=split, transform=display_transform, convert_rgb=True)
    print(dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                         drop_last=False)

    with torch.no_grad():
        for i, (images, target, _) in enumerate(tqdm(loader)):
            img = np.clip(images.cpu().numpy(), 0, 1)  # [0, 1]
            img = img.transpose(0, 2, 3, 1)
            img = (img * 255).astype(np.uint8)
            img = img.squeeze()

            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            axs.imshow(img, cmap='gray')
            axs.axis("off")
            plt.show()

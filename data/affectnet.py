#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://github.com/yaoing/dan
https://github.com/ElenaRyumina/EMO-AffectNetModel
https://github.com/PanosAntoniadis/emotion-gcn
"""

__author__ = "GZ"

import os
import sys
from shutil import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if sys.platform == 'win32':
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


# def convert_affectnet(label_file, save_dir):
# 	df = pd.read_csv(label_file)
#
# 	for i in range(8):
# 	    for j in ['train','val']:
# 	        os.makedirs(os.path.join(save_dir, "AffectNet", j, i), exist_ok=True)
#
# 	for i, row in df.iterrows():
# 	    p = row['phase']
# 	    l = row['label']
# 	    copy(row['img_path'], os.path.join(save_dir, "AffectNet", p, l))
#
# 	print('convert done.')
#
#
# def get_AffectNet(root, split, transform, num_class=7):
# 	data_dir = os.path.join(root, split)
# 	dataset = datasets.ImageFolder(data_dir, transform=transform)
# 	if num_class == 7:  # ignore the 8-th class
# 		idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] != 7]
# 		dataset = data.Subset(dataset, idx)
# 	return dataset


def generate_affectnet(img_dir, label_dir, split, save_dir, num_class=7):
	assert split in ["train", "val"]
	label_file = "training.csv" if split == "train" else "validation.csv"
	head_list = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks',
	             'expression', 'valence', 'arousal']
	dict_name_labels = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

	df_data_raw = pd.read_csv(os.path.join(label_dir, label_file))
	df_data_raw.expression = pd.to_numeric(df_data_raw.expression, errors='coerce').fillna(100).astype('int64')

	df_data = df_data_raw[df_data_raw['expression'] < num_class]

	for label in range(num_class):
		os.makedirs(os.path.join(save_dir, split, str(label)), exist_ok=True)

	file_notfound = []
	for i, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
		label = row['expression']
		img_file = os.path.join(img_dir, row['subDirectory_filePath'])

		if os.path.isfile(img_file):
			copy(img_file, os.path.join(save_dir, split, str(label)))
		else:
			file_notfound.append(img_file)

	# 2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg
	print(file_notfound)


if __name__ == '__main__':
	import torch
	import torch.utils.data as data
	from torchvision import transforms, datasets
	from data.base_dataset import ImageFolderInstance
	from data.sampler import DistributedImbalancedSampler, DistributedSamplerWrapper, ImbalancedDatasetSampler

	# label_file = "../data/FER/AffectNet/affectnet.csv"
	# save_dir = "../data/FER"
	# convert_affectnet(label_file, save_dir)

	img_dir = "../data/FER/AffectNet/Manually_Annotated_Images"
	label_dir = '../data/FER/AffectNet/Manually_Annotated_file_lists'
	split = "train"
	save_dir = "../data/FER/AffectNet_subset"
	# generate_affectnet(img_dir, label_dir, split, save_dir)

	data_root = save_dir
	display_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])

	# dataset = get_AffectNet(data_root, split, display_transform, num_class=7)
	data_dir = os.path.join(data_root, split)
	dataset = ImageFolderInstance(data_dir, transform=display_transform)
	print(dataset)

	train_percent = 0.1
	if train_percent < 1.0:
		num_subset = int(len(dataset) * train_percent)
		indices = torch.randperm(len(dataset))[:num_subset]
		indices = indices.tolist()
		dataset = torch.utils.data.Subset(dataset, indices)
		print("Sub train_dataset:\n{}".format(len(dataset)))

	sampler = ImbalancedDatasetSampler(dataset)
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

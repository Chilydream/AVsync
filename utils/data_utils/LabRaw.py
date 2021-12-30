import torch
import numpy as np
import random
import pdb
import os
import threading
import time
from queue import Queue
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils.GetDataFromFile import get_mfcc, get_frame_tensor, get_wav, get_frame_and_wav


class LabDataset(Dataset):
	def __init__(self, dataset_file, seq_len, resolution, max_size):
		super(LabDataset, self).__init__()
		self.dataset_file_name = dataset_file
		self.seq_len = seq_len
		self.resolution = resolution
		self.max_size = max_size
		self.file_list = []
		# self.talk_flag = []
		# todo:　静音数据集
		self.nfile = 0

		with open(dataset_file) as fr:
			for idx, line in enumerate(fr.readlines()):
				items = line.strip().split('\t')
				if len(items)==2:
					is_talk, filename = items
					if is_talk != '0':
						self.file_list.append(filename)
				elif len(items)==1:
					filename = items[0]
					self.file_list.append(filename)
		self.nfile = len(self.file_list)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		img_tensor, wav_match, wav_mismatch = get_frame_and_wav(mp4_name,
		                                                        seq_len=self.seq_len,
		                                                        resolution=self.resolution)
		return img_tensor, wav_match, wav_mismatch

	def __len__(self):
		if self.max_size<=0:
			return self.nfile
		return min(self.max_size, self.nfile)


class LabDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, seq_len, resolution, is_train=True, max_size=0):
		# todo: max_size不为0时，似乎不会进行shuffle，有待解决
		self.dataset_file = dataset_file
		self.dataset = LabDataset(dataset_file=dataset_file,
		                          seq_len=seq_len,
		                          resolution=resolution,
		                          max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(dataset=self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

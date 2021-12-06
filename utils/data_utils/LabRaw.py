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

from utils.GetDataFromFile import get_mfcc, get_frame_tensor, get_wav


class LabDataset(Dataset):
	def __init__(self, dataset_file, max_size, seq_len=0):
		super(LabDataset, self).__init__()
		if isinstance(dataset_file, str):
			self.dataset_file_name = [dataset_file]
		else:
			self.dataset_file_name = dataset_file
		self.seq_len = seq_len
		self.max_size = max_size
		self.silent_file = []
		self.speak_file = []

		for metafile in self.dataset_file_name:
			with open(metafile) as fr:
				for idx, line in enumerate(fr.readlines()):
					items = line.strip().split('\t')
					if items[0] == '0':
						self.silent_file.append(items[1])
					else:
						self.speak_file.append(items[1])
		self.n_silent = len(self.silent_file)
		self.n_speak = len(self.speak_file)
		self.nfile = self.n_silent+self.n_speak

	def __getitem__(self, item):
		if item<self.n_silent:
			mp4_name = self.silent_file[item]
			is_speak = -1
		else:
			mp4_name = self.speak_file[item-self.n_silent]
			is_speak = 1
		frame_tensor = get_frame_tensor(mp4_name, self.seq_len, resolution=256)
		return frame_tensor, is_speak

	def __len__(self):
		if self.max_size<=0:
			return self.nfile
		return min(self.nfile, self.max_size)


class LabDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, seq_len=16, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LabDataset(dataset_file, max_size, seq_len)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

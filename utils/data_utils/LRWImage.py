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
from utils.data_utils.LRWRaw import LRWDataset


class LRWImageDataset(LRWDataset):
	def __init__(self, dataset_file, seq_len, resolution, max_size):
		super().__init__(dataset_file=dataset_file, n_mfcc=-1,
		                 seq_len=seq_len,
		                 resolution=resolution,
		                 max_size=max_size)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		frame_tensor = get_frame_tensor(filename=mp4_name,
		                                seq_len=self.seq_len,
		                                resolution=self.resolution)
		return frame_tensor, self.id2wid[item]


class LRWImageDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers,
	             seq_len=0, resolution=0, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LRWImageDataset(dataset_file=dataset_file,
		                               seq_len=seq_len,
		                               resolution=resolution,
		                               max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

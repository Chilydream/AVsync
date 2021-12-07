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


class LRWImageLmkDataset(LRWDataset):
	def __init__(self, dataset_file, seq_len, max_size):
		super(LRWImageLmkDataset, self).__init__(dataset_file=dataset_file, n_mfcc=-1,
		                                         seq_len=seq_len,
		                                         resolution=0,
		                                         max_size=max_size)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		lmk_name = mp4_name[:-3]+'lmk'
		lmk_tensor = torch.load(lmk_name)
		return lmk_tensor, self.id2wid[item]


class LRWImageLmkDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers,
	             seq_len=0, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LRWImageLmkDataset(dataset_file=dataset_file,
		                                  seq_len=seq_len,
		                                  max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

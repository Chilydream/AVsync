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


class LRWAudioDataset(LRWDataset):
	def __init__(self, dataset_file, n_mfcc, max_size):
		super(LRWAudioDataset, self).__init__(dataset_file, n_mfcc, seq_len=0, resolution=-1,
		                                      max_size=max_size)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		wav_name = mp4_name[:-3]+'wav'
		if self.n_mfcc == 0:
			wav_tensor = get_wav(wav_name)
		elif self.n_mfcc<0:
			return wav_name, self.id2wid[item]
		else:
			wav_tensor = get_mfcc(wav_name, self.n_mfcc)
		return wav_tensor, self.id2wid[item]


class LRWAudioDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, n_mfcc, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LRWAudioDataset(dataset_file, n_mfcc, max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

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
from utils.data_utils.LRWAudio import LRWAudioDataset
from utils.data_utils.LRWImage import LRWImageDataset
from torch.utils.data.distributed import DistributedSampler


class LRWImageTripletDataset(LRWImageDataset):
	def __init__(self, dataset_file, seq_len, resolution, max_size):
		super(LRWImageTripletDataset, self).__init__(dataset_file=dataset_file,
		                                             seq_len=seq_len,
		                                             resolution=resolution,
		                                             max_size=max_size)

	def get_rand_id_from_wid(self, wid):
		start_id = self.word_accumulate[wid]
		end_id = self.word_accumulate[wid+1]
		rand_id = np.random.randint(start_id, end_id)
		return rand_id

	def __getitem__(self, item):
		a_mp4name = self.file_list[item]

		p_wid = self.id2wid[item]
		p_id = self.get_rand_id_from_wid(p_wid)
		p_mp4name = self.file_list[p_id]

		while True:
			n_wid = random.randint(0, self.nword-1)
			if n_wid != p_wid:
				break
		n_id = self.get_rand_id_from_wid(n_wid)
		n_mp4name = self.file_list[n_id]

		a_frame_tensor = get_frame_tensor(filename=a_mp4name, seq_len=self.seq_len, resolution=self.resolution)
		p_frame_tensor = get_frame_tensor(filename=p_mp4name, seq_len=self.seq_len, resolution=self.resolution)
		n_frame_tensor = get_frame_tensor(filename=n_mp4name, seq_len=self.seq_len, resolution=self.resolution)
		return a_frame_tensor, p_frame_tensor, n_frame_tensor, p_wid, n_wid


class LRWImageTripletDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, seq_len=0, resolution=0,
	             is_train=True, max_size=0, distributed=False):
		self.dataset_file = dataset_file
		self.dataset = LRWImageTripletDataset(dataset_file=dataset_file,
		                                      seq_len=seq_len,
		                                      resolution=resolution,
		                                      max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		if distributed:
			super().__init__(self.dataset, batch_size=batch_size,
			                 drop_last=True, num_workers=num_workers, pin_memory=False,
			                 sampler=DistributedSampler(self.dataset))
		else:
			super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
			                 drop_last=True, num_workers=num_workers, pin_memory=False,)

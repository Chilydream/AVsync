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


class LRWImageLmkTripletDataset(LRWDataset):
	def __init__(self, dataset_file, seq_len, max_size):
		super().__init__(dataset_file=dataset_file, n_mfcc=-1,
		                 seq_len=seq_len,
		                 resolution=0,
		                 max_size=max_size)

	def get_rand_id_from_wid(self, wid):
		start_id = self.word_accumulate[wid]
		end_id = self.word_accumulate[wid+1]
		rand_id = np.random.randint(start_id, end_id)
		return rand_id

	def __getitem__(self, item):
		a_lmkname = self.file_list[item][:-3]+'lmk'

		p_wid = self.id2wid[item]
		p_id = self.get_rand_id_from_wid(p_wid)
		p_lmkname = self.file_list[p_id][:-3]+'lmk'

		while True:
			n_wid = random.randint(0, self.nword-1)
			if n_wid != p_wid:
				break
		n_id = self.get_rand_id_from_wid(n_wid)
		n_lmkname = self.file_list[n_id][:-3]+'lmk'

		a_lmk_tensor = torch.load(a_lmkname)
		p_lmk_tensor = torch.load(p_lmkname)
		n_lmk_tensor = torch.load(n_lmkname)
		a_lmk_tensor = torch.flatten(a_lmk_tensor[:, 48:68, :], start_dim=1)
		p_lmk_tensor = torch.flatten(p_lmk_tensor[:, 48:68, :], start_dim=1)
		n_lmk_tensor = torch.flatten(n_lmk_tensor[:, 48:68, :], start_dim=1)
		return a_lmk_tensor, p_lmk_tensor, n_lmk_tensor, p_wid, n_wid


class LRWImageLmkTripletDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers,
	             seq_len=0, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LRWImageLmkTripletDataset(dataset_file=dataset_file,
		                                         seq_len=seq_len,
		                                         max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

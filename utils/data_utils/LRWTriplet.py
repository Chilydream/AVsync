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


class LRWTripletDataset(LRWDataset):
	def __init__(self, dataset_file, nmfcc, resolution, seq_len, max_size):
		super().__init__(dataset_file=dataset_file, n_mfcc=nmfcc,
		                 seq_len=seq_len,
		                 resolution=resolution,
		                 max_size=max_size)

	def get_rand_id_from_wid(self, wid):
		start_id = self.word_accumulate[wid]
		end_id = self.word_accumulate[wid+1]
		rand_id = np.random.randint(start_id, end_id)
		return rand_id

	def __getitem__(self, item):
		# todo: seq_len还没有用上
		# todo: 获取item也需要修改
		# todo: 取关键点的方法也需要普适性
		a_mp4name = self.file_list[item]
		a_wavname = a_mp4name[:-3]+'wav'
		a_lmkname = a_mp4name[:-3]+'lmk'

		p_wid = self.id2wid[item]
		p_id = self.get_rand_id_from_wid(p_wid)
		p_mp4name = self.file_list[p_id]
		p_wavname = p_mp4name[:-3]+'wav'
		p_lmkname = p_mp4name[:-3]+'lmk'

		while True:
			n_wid = random.randint(0, self.nword-1)
			if n_wid != p_wid:
				break
		n_id = self.get_rand_id_from_wid(n_wid)
		n_mp4name = self.file_list[n_id]
		n_wavname = n_mp4name[:-3]+'wav'
		n_lmkname = n_mp4name[:-3]+'lmk'

		if self.n_mfcc==0:
			a_wav = get_wav(a_wavname)
			p_wav = get_wav(p_wavname)
			n_wav = get_wav(n_wavname)
		elif self.n_mfcc<0:
			a_wav = a_wavname
			p_wav = p_wavname
			n_wav = n_wavname
		else:
			a_wav = get_mfcc(a_wavname, self.n_mfcc)
			p_wav = get_mfcc(p_wavname, self.n_mfcc)
			n_wav = get_mfcc(n_wavname, self.n_mfcc)

		if self.resolution>=0:
			a_img = get_frame_tensor(a_mp4name, resolution=self.resolution)
			p_img = get_frame_tensor(p_mp4name, resolution=self.resolution)
			n_img = get_frame_tensor(n_mp4name, resolution=self.resolution)
		else:
			a_img = torch.load(a_lmkname)
			p_img = torch.load(p_lmkname)
			n_img = torch.load(n_lmkname)
			a_img = torch.flatten(a_img[:, 48:68, :], start_dim=1)
			p_img = torch.flatten(p_img[:, 48:68, :], start_dim=1)
			n_img = torch.flatten(n_img[:, 48:68, :], start_dim=1)
		return a_wav, p_wav, n_wav, a_img, p_img, n_img, p_wid, n_wid


class LRWTripletDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers,
	             n_mfcc, resolution,
	             seq_len=0, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LRWTripletDataset(dataset_file=dataset_file,
		                                 nmfcc=n_mfcc,
		                                 resolution=resolution,
		                                 seq_len=seq_len,
		                                 max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

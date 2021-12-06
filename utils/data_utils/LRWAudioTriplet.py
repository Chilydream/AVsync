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


class LRWAudioTripletDataset(LRWAudioDataset):
	def __init__(self, dataset_file, n_mfcc, max_size):
		super(LRWAudioTripletDataset, self).__init__(dataset_file, n_mfcc, max_size)

	def get_rand_id_from_wid(self, wid):
		start_id = self.word_accumulate[wid]
		end_id = self.word_accumulate[wid+1]
		rand_id = np.random.randint(start_id, end_id)
		return rand_id

	def __getitem__(self, item):
		a_wavname = self.file_list[item][:-3]+'wav'

		p_wid = self.id2wid[item]
		p_id = self.get_rand_id_from_wid(p_wid)
		p_wavname = self.file_list[p_id][:-3]+'wav'

		while True:
			n_wid = random.randint(0, self.nword-1)
			if n_wid != p_wid:
				break
		n_id = self.get_rand_id_from_wid(n_wid)
		n_wavname = self.file_list[n_id][:-3]+'wav'

		if self.n_mfcc == 0:
			a_wav_tensor = get_wav(a_wavname)
			p_wav_tensor = get_wav(p_wavname)
			n_wav_tensor = get_wav(n_wavname)
		elif self.n_mfcc<0:
			return a_wavname, p_wavname, n_wavname, p_wid, n_wid
		else:
			a_wav_tensor = get_mfcc(a_wavname, self.n_mfcc)
			p_wav_tensor = get_mfcc(p_wavname, self.n_mfcc)
			n_wav_tensor = get_mfcc(n_wavname, self.n_mfcc)
		return a_wav_tensor, p_wav_tensor, n_wav_tensor, p_wid, n_wid


class LRWAudioTripletDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, n_mfcc, is_train=True, max_size=0):
		self.dataset_file = dataset_file
		self.dataset = LRWAudioTripletDataset(dataset_file, n_mfcc, max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

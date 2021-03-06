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


class LRWDataset(Dataset):
	def __init__(self, dataset_file, n_mfcc, seq_len, resolution, max_size):
		super(LRWDataset, self).__init__()
		self.dataset_file_name = dataset_file
		self.n_mfcc = n_mfcc
		self.seq_len = seq_len
		self.resolution = resolution
		self.max_size = max_size
		self.word_list = []
		self.word_accumulate = []
		# word_accumulate[0] = 0
		# 表示0-999都是第 0个单词
		# word_accumulate[1] = 1000
		# 表示1000-1958都是第 1个单词
		# word_accumulate[2] = 1959
		# word_accumulate[n-1] = 487766
		# 表示 487766到 488766都是第 n-1个单词
		# word_accumulate[n] = 488766
		# wid从 0到 n-1，一共有 n个单词
		self.file_list = []
		self.nfile = 0
		self.nword = 0
		self.id2wid = []

		with open(dataset_file) as fr:
			last_word = None
			wid = -1
			for idx, line in enumerate(fr.readlines()):
				word, filename = line.strip().split('\t')
				# wavname = filename[:-3]+'wav'
				# if not os.path.exists(filename) or not os.path.exists(wavname):
				# 	continue
				self.file_list.append(filename)
				if last_word != word:
					wid += 1
					if last_word is not None:
						self.word_list.append(word)
					self.word_accumulate.append(idx)
					last_word = word
				self.id2wid.append(wid)
		self.nword = wid+1
		self.nfile = len(self.file_list)
		self.word_accumulate.append(self.nfile)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		wav_name = mp4_name[:-3]+'wav'
		lmk_name = mp4_name[:-3]+'lmk'
		if self.n_mfcc == 0:
			a_wav = get_wav(filename=wav_name)
		elif self.n_mfcc<0:
			a_wav = item
		else:
			a_wav = get_mfcc(filename=wav_name,
			                 n_mfcc=self.n_mfcc)

		if self.resolution>=0:
			a_img = get_frame_tensor(filename=mp4_name,
			                         seq_len=self.seq_len,
			                         resolution=self.resolution)
		else:
			a_img = torch.load(lmk_name)
			a_img = torch.flatten(a_img[:, 48:68, :], start_dim=1)
		return a_wav, a_img, self.id2wid[item]

	def get_rand_id_from_wid(self, wid):
		start_id = self.word_accumulate[wid]
		end_id = self.word_accumulate[wid+1]
		rand_id = np.random.randint(start_id, end_id)
		return rand_id

	def __len__(self):
		if self.max_size<=0:
			return self.nfile
		return min(self.max_size, self.nfile)


class LRWDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, n_mfcc, seq_len, resolution, is_train=True, max_size=0):
		# todo: max_size不为0时，似乎不会进行shuffle，有待解决
		self.dataset_file = dataset_file
		self.dataset = LRWDataset(dataset_file=dataset_file,
		                          n_mfcc=n_mfcc,
		                          seq_len=seq_len,
		                          resolution=resolution,
		                          max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(dataset=self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

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


class LabLmkDataset(Dataset):
	def __init__(self, dataset_file, seq_len, max_size):
		super(LabLmkDataset, self).__init__()
		self.dataset_file_name = dataset_file
		self.seq_len = seq_len
		self.max_size = max_size
		self.file_list = []
		# self.talk_flag = []
		# todo:　静音数据集
		self.nfile = 0

		with open(dataset_file) as fr:
			for idx, line in enumerate(fr.readlines()):
				is_talk, filename = line.strip().split('\t')
				if is_talk != '0':
					self.file_list.append(filename)
		self.nfile = len(self.file_list)

	def __getitem__(self, item):
		mp4name = self.file_list[item]
		lmkname = mp4name[:-3]+'lmk'
		wavname = mp4name[:-3]+'wav'
		all_lmk = torch.load(lmkname)
		all_wav = get_wav(wavname)
		min_len = min(all_lmk.shape[0], len(all_wav)*25/16000)
		start_pos = np.random.randint(0, min_len-30-self.seq_len)

		avg_lmk = torch.mean(all_lmk, dim=(0, 1))
		var_lmk = torch.var(all_lmk, dim=(0, 1))
		all_lmk = (all_lmk-avg_lmk)/var_lmk
		lmk_tensor = all_lmk[start_pos:start_pos+self.seq_len]
		lmk_tensor = torch.flatten(lmk_tensor[:, 48:68, :], start_dim=1)

		wav_match = all_wav[int(start_pos*16000/25):
		                    int((start_pos+self.seq_len)*16000/25)]
		wav_mismatch = all_wav[int((start_pos+25)*16000/25):
		                       int((start_pos+25+self.seq_len)*16000/25)]

		return lmk_tensor, wav_match, wav_mismatch

	def __len__(self):
		if self.max_size<=0:
			return self.nfile
		return min(self.max_size, self.nfile)


class LabLmkDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers, seq_len, is_train=True, max_size=0):
		# todo: max_size不为0时，似乎不会进行shuffle，有待解决
		self.dataset_file = dataset_file
		self.dataset = LabLmkDataset(dataset_file=dataset_file,
		                             seq_len=seq_len,
		                             max_size=max_size)
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.max_size = max_size
		super().__init__(dataset=self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

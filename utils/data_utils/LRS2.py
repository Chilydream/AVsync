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

from utils.GetDataFromFile import get_mfcc, get_frame_tensor, get_wav, get_frame_and_wav, get_frame_and_wav_cv2


class LRS2Dataset(Dataset):
	def __init__(self, dataset_file, seq_len, resolution, max_size):
		super(LRS2Dataset, self).__init__()
		self.dataset_file_name = dataset_file
		self.seq_len = seq_len
		self.resolution = resolution
		self.max_size = max_size
		self.file_list = []
		self.nfile = 0

		with open(dataset_file) as fr:
			for idx, line in enumerate(fr.readlines()):
				items = line.strip().split('\t')
				if len(items) == 2:
					is_talk, filename = items
					if is_talk != '0':
						self.file_list.append(filename)
				elif len(items) == 1:
					filename = items[0]
					if os.path.exists(filename):
						self.file_list.append(filename)
		self.nfile = len(self.file_list)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		wav_name = mp4_name[:-3]+'wav'
		npy_name = mp4_name[:-3]+'npy'
		wav_tensor = get_wav(wav_name)
		npy_tensor = np.load(npy_name)

		return npy_tensor, wav_tensor

	def __len__(self):
		if self.max_size<=0:
			return self.nfile
		return min(self.max_size, self.nfile)

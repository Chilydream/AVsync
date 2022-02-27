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

from utils.GetDataFromFile import get_frame_and_wav_cv2


class LabDataset(Dataset):
	def __init__(self, dataset_file,
	             tgt_frame_num, tgt_fps, resolution,
	             wav_hz,
	             avspeech_flag=False):
		super(LabDataset, self).__init__()
		self.dataset_file_name = dataset_file
		self.tgt_frame_num = tgt_frame_num
		self.tgt_fps = tgt_fps
		self.wav_hz = wav_hz
		self.resolution = resolution
		self.file_list = []
		self.length_list = []
		self.nfile = 0

		with open(dataset_file) as fr:
			for idx, line in enumerate(fr.readlines()):
				items = line.strip().split('\t')
				if avspeech_flag:
					filename = items[0]
					if os.path.exists(filename):
						self.file_list.append(filename)
						# SeTlgy7GVXU_004.605000-009.243000.mp4
						# 012345678901234567890123456789012
						time_length = float(filename[23:33])-float(filename[12:22])
						self.length_list.append(time_length)
				else:
					if len(items) == 2:
						is_talk, filename = items
						if is_talk != '0':
							self.file_list.append(filename)
							self.length_list.append(0)
					elif len(items) == 1:
						filename = items[0]
						if os.path.exists(filename):
							self.file_list.append(filename)
							self.length_list.append(0)

		self.nfile = len(self.file_list)
		print(self.nfile)

	def __getitem__(self, item):
		mp4_name = self.file_list[item]
		img_tensor, wav_tensor = get_frame_and_wav_cv2(filename=mp4_name,
		                                               tgt_frame_num=self.tgt_frame_num,
		                                               tgt_fps=self.tgt_fps,
		                                               resolution=self.resolution,
		                                               wav_hz=self.wav_hz,
		                                               total_time=self.length_list[item])
		return img_tensor, wav_tensor

	def __len__(self):
		return self.nfile


class LabDataLoader(DataLoader):
	def __init__(self, dataset_file, batch_size, num_workers,
	             tgt_frame_num, tgt_fps, resolution,
	             wav_hz=16000,
	             avspeech_flag=False,
	             is_train=True):
		self.dataset_file = dataset_file
		self.dataset = LabDataset(dataset_file=dataset_file,
		                          tgt_frame_num=tgt_frame_num,
		                          tgt_fps=tgt_fps,
		                          resolution=resolution,
		                          wav_hz=wav_hz,
		                          avspeech_flag=avspeech_flag
		                          )
		self.num_workers = num_workers
		self.batch_size = batch_size
		super().__init__(dataset=self.dataset, shuffle=is_train, batch_size=batch_size,
		                 drop_last=True, num_workers=num_workers)

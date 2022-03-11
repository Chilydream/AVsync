import librosa
import torch
import numpy as np
import random
import pdb
import os
import cv2
import threading
import time
from queue import Queue
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils import audio
from utils.GetDataFromFile import make_image_square


class LRS2FaceDataset(Dataset):
	def __init__(self, dataset_file, tgt_frame_num, img_size):
		super(LRS2FaceDataset, self).__init__()
		self.dataset_file_name = dataset_file
		self.seq_len = tgt_frame_num
		self.img_size = img_size
		self.file_list = []
		self.face_num_list = []
		self.nfile = 0

		self.sample_rate = 16000
		self.syncnet_mel_step_size = 16

		with open(dataset_file) as fr:
			for idx, line in enumerate(fr.readlines()):
				items = line.strip().split('\t')
				filename, face_num = items
				self.file_list.append(filename)
				self.face_num_list.append(int(face_num))
		self.nfile = len(self.file_list)

	def get_rand_start_frame(self, frame_num):
		match_id = np.random.randint(0, frame_num-self.seq_len)
		match_end = match_id+self.seq_len
		if match_id<=self.seq_len:
			mismatch_id = np.random.randint(match_end, frame_num-self.seq_len)
		elif match_end+self.seq_len>=frame_num:
			mismatch_id = np.random.randint(0, match_id-self.seq_len)
		else:
			mismatch_id0 = np.random.randint(0, match_id-self.seq_len)
			mismatch_id1 = np.random.randint(match_end, frame_num-self.seq_len)
			mismatch_id = mismatch_id0 if np.random.randint(0, 2) == 0 else mismatch_id1

		return match_id, mismatch_id

	def get_frame_id(self, frame):
		return int(os.path.basename(frame).split('.')[0])

	def get_window(self, start_frame):
		start_id = self.get_frame_id(start_frame)
		vidname = os.path.dirname(start_frame)

		window_fnames = []
		for frame_id in range(start_id, start_id+self.seq_len):
			frame = os.path.join(vidname, '{}.npy'.format(frame_id))
			if not os.path.isfile(frame):
				# todo: 提前去掉不全面的数据，减少读取时间
				return None
			window_fnames.append(frame)
		return window_fnames

	def __getitem__(self, item):
		while 1:
			idx = random.randint(0, len(self.file_list)-1)
			vidname, frame_num = self.file_list[idx], self.face_num_list[idx]

			match_id, mismatch_id = self.get_rand_start_frame(frame_num)
			img_npy = os.path.join(vidname, f'{match_id}.npy')
			wrong_img_npy = os.path.join(vidname, f'{mismatch_id}.npy')

			if random.choice([True, False]):
				y = torch.ones(1).float()
				chosen = img_npy
			else:
				y = torch.zeros(1).float()
				chosen = wrong_img_npy

			window_fnames = self.get_window(chosen)
			if window_fnames is None:
				print(match_id, mismatch_id)
				print(img_npy)
				print(wrong_img_npy)
				print(chosen)
				print('get window error')
				continue

			window = []
			all_read = True
			for npy_name in window_fnames:
				try:
					img = np.load(npy_name)
					if self.img_size != 0:
						img = make_image_square(img)
						img = cv2.resize(img, (self.img_size, self.img_size))
				except Exception as e:
					print('in loading npy', e)
					all_read = False
					break

				window.append(img)

			if not all_read:
				print(chosen)
				print('img read error')
				continue

			try:
				wavpath = os.path.join(vidname, "audio.wav")
				wav = audio.load_wav(wavpath, self.sample_rate)

				orig_mel = audio.melspectrogram(wav).T
			except Exception as e:
				print(chosen)
				print(e)
				print('wav loading error')
				continue

			mel = self.crop_audio_window(orig_mel.copy(), img_npy)

			if mel.shape[0] != self.syncnet_mel_step_size:
				print(chosen)
				print('wav crop error')
				continue

			# H x W x 3 * T
			x = np.concatenate(window, axis=2)/255.
			x = x.transpose(2, 0, 1)
			x = x[:, x.shape[1]//2:]

			x = torch.FloatTensor(x)
			mel = torch.FloatTensor(mel.T).unsqueeze(0)

			return x, mel, y

	def __len__(self):
		return self.nfile

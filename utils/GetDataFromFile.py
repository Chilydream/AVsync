import math
import os

import torch
import numpy as np
import cv2
import librosa
from moviepy.editor import VideoFileClip


def get_mfcc(filename: str, n_mfcc):
	y, sr = librosa.load(filename, sr=None)
	mfcc_seq = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=len(y)//28)
	mfcc_seq = np.transpose(mfcc_seq, (1, 0))
	# hop_length=730是为了最终返回的特征长度是29，可以和视频特征相匹配
	# windows 上预处理出来的 wav就是 690

	return mfcc_seq


def get_wav(filename):
	y, sr = librosa.load(filename, sr=16000)
	return y


def get_frame_and_wav(filename, seq_len=29, video_fps=25, resolution=0):
	video_file_clip = VideoFileClip(filename)
	audio_file_clip = video_file_clip.audio
	whole_length = video_file_clip.duration
	if whole_length<(seq_len-1)/video_fps:
		raise ValueError(f'要求视频时长不小于{seq_len/video_fps}秒，但是文件“{filename}”只有{whole_length}秒')

	seq_duration = seq_len/video_fps
	start_time = np.random.uniform(0, whole_length-1-seq_duration)
	start_time = min(start_time, 2)
	video_file_clip = video_file_clip.to_RGB()
	image_list = []
	for i in range(seq_len):
		image = video_file_clip.make_frame(start_time+i/video_fps)
		if resolution != 0:
			image = make_image_square(image)
			image = cv2.resize(image, (resolution, resolution))
		image_list.append(image)

	wavname = filename[:-3]+'wav'
	if os.path.exists(wavname):
		wav_array = get_wav(wavname)
	else:
		wav_array = audio_file_clip.to_soundarray(fps=16000)[:, 0]
	# wav_offset = 1
	wav_match_start = int(start_time*16000)
	wav_match = wav_array[wav_match_start:
	                      wav_match_start+int(seq_len*16000/video_fps)]
	wav_match = torch.FloatTensor(wav_match)
	# wav_mismatch_start = int((start_time+wav_offset)*16000)
	# wav_mismatch = wav_array[wav_mismatch_start:wav_mismatch_start+int(seq_len*16000/video_fps)]
	# wav_mismatch = torch.FloatTensor(wav_mismatch)

	video_file_clip.close()
	im = np.stack(image_list, axis=3)
	# stack操作后 im的形状是（256,256,3,29）
	im = np.transpose(im, (3, 2, 0, 1))
	# im的形状是（29,3,256,256）
	im_tensor = torch.FloatTensor(im)

	# return im_tensor, wav_match, wav_mismatch
	return im_tensor, wav_match


def get_frame_and_wav_cv2(filename, seq_len=15, tgt_fps=15, resolution=0):
	if isinstance(resolution, int):
		resolution = (resolution, resolution)
	cap = cv2.VideoCapture(filename)
	src_fps = cap.get(cv2.CAP_PROP_FPS)
	src_fps = int(src_fps+0.5)

	start_frame = np.random.randint(0, tgt_fps)
	# start_frame = 0
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	start_time = start_frame*1.0/tgt_fps

	image_list = []
	image_cnt = 0
	raw_frame_num = int(seq_len*src_fps/tgt_fps)
	while True:
		image_cnt += 1
		if image_cnt>raw_frame_num != 0:
			break
		ret, image = cap.read()
		if image is None:
			break

		if resolution != (0, 0):
			image = make_image_square(image)
			image = cv2.resize(image, resolution)
		image_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	if len(image_list)<seq_len:
		print(f'{filename} len(image_list){len(image_list)}, raw_num{raw_frame_num}, {src_fps}')
	cap.release()
	im = np.stack(image_list, axis=3)
	# stack操作后 im的形状是（256,256,3,29）
	im = np.transpose(im, (3, 2, 0, 1))
	# im的形状是（29,3,256,256）
	im_tensor = torch.FloatTensor(im)
	if raw_frame_num != seq_len and seq_len != 0:
		frac_ratio = src_fps/tgt_fps
		new_idx = list(map(lambda i: int(i*frac_ratio), range(seq_len)))
		im_tensor = im_tensor[new_idx, ...]
	wavname = filename[:-3]+'wav'
	wav_array = get_wav(wavname)
	# wav_offset = 1
	wav_start = int(start_time*16000)
	wav_tensor = wav_array[wav_start:wav_start+int(seq_len*16000/tgt_fps)]
	wav_tensor = torch.FloatTensor(wav_tensor)
	return im_tensor, wav_tensor


def make_image_square(img):
	s = max(img.shape[0:2])
	f = np.zeros((s, s, 3), np.uint8)
	ax, ay = (s-img.shape[1])//2, (s-img.shape[0])//2
	f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
	return f


def get_frame_moviepy(filename, seq_len=29, fps=25, resolution=0):
	video_file_clip = VideoFileClip(filename)
	video_file_clip = video_file_clip.to_RGB()
	image_list = []
	video_fps = fps
	for i in range(seq_len):
		image = video_file_clip.make_frame(i/video_fps)
		if resolution != 0:
			image = make_image_square(image)
			image = cv2.resize(image, (resolution, resolution))
		image_list.append(image)
	video_file_clip.close()
	im = np.stack(image_list, axis=3)
	# stack操作后 im的形状是（256,256,3,29）
	im = np.transpose(im, (3, 2, 0, 1))
	# im的形状是（29,3,256,256）
	im_tensor = torch.FloatTensor(im)

	return im_tensor


def get_frame_tensor(filename, seq_len=0, resolution=0, tgt_fps=25):
	# todo: 所有的LRW视频都是25fps,一共29帧,分辨率为256*256
	cap = cv2.VideoCapture(filename)
	src_fps = cap.get(cv2.CAP_PROP_FPS)
	if src_fps<tgt_fps:
		# ques：当原fps小于25要怎么处理？
		src_fps = tgt_fps
	# cap.set(cv2.CAP_PROP_FPS, 25)
	if seq_len>0:
		# video_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		start_frame = np.random.randint(0, 2*tgt_fps)
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

	image_list = []
	image_cnt = 0
	raw_frame_num = int(seq_len*src_fps/tgt_fps)
	while True:
		image_cnt += 1
		if image_cnt>raw_frame_num != 0:
			break
		ret, image = cap.read()
		if image is None:
			break

		if resolution != 0:
			image = make_image_square(image)
			image = cv2.resize(image, (resolution, resolution))
		image_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	cap.release()
	if len(image_list)<29:
		raise ValueError(f'{filename} image_list 有问题，长度只有{len(image_list)}，开始位置{start_frame}')
	im = np.stack(image_list, axis=3)
	# stack操作后 im的形状是（256,256,3,29）
	im = np.transpose(im, (3, 2, 0, 1))
	# im的形状是（29,3,256,256）
	if raw_frame_num != seq_len and seq_len != 0:
		frac_ratio = src_fps/tgt_fps
		new_idx = list(map(lambda i: int(i*frac_ratio), range(math.ceil(raw_frame_num/frac_ratio))))
		if len(new_idx) != 29:
			print(seq_len, raw_frame_num, src_fps, tgt_fps)
			raise ValueError(f'{filename}图像长度不是29')
		im = im[new_idx]
	im_tensor = torch.FloatTensor(im)

	return im_tensor

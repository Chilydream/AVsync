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

	# start_time = np.random.randint(0, int(whole_length-1-(seq_len-1)/video_fps))
	start_time = np.random.uniform(0, whole_length-2-(seq_len-1)/video_fps)
	video_file_clip = video_file_clip.to_RGB()
	image_list = []
	video_fps = video_fps
	for i in range(seq_len):
		image = video_file_clip.make_frame(start_time+i/video_fps)
		if resolution != 0:
			image = make_image_square(image)
			image = cv2.resize(image, (resolution, resolution))
		image_list.append(image)

	wav_array = audio_file_clip.to_soundarray(fps=16000)[:, 0]
	wav_match = wav_array[int(start_time*16000):
	                      int((start_time+(seq_len-1)/video_fps)*16000)]
	wav_mismatch = wav_array[int((start_time+2)*16000):
	                         int((start_time+2+(seq_len-1)/video_fps)*16000)]
	wav_match = torch.FloatTensor(wav_match)
	wav_mismatch = torch.FloatTensor(wav_mismatch)

	video_file_clip.close()
	im = np.stack(image_list, axis=3)
	# stack操作后 im的形状是（256,256,3,29）
	im = np.transpose(im, (3, 2, 0, 1))
	# im的形状是（29,3,256,256）
	im_tensor = torch.FloatTensor(im)

	return im_tensor, wav_match, wav_mismatch


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


def get_frame_tensor(filename, seq_len=0, resolution=0):
	# todo: 所有的LRW视频都是25fps,一共29帧,分辨率为256*256
	cap = cv2.VideoCapture(filename)
	cap.set(cv2.CAP_PROP_FPS, 25)
	if seq_len>0:
		video_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		start_frame = np.random.randint(0, video_len-5*seq_len)
		start_frame = max(0, start_frame)
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

	image_list = []
	image_cnt = 0
	while True:
		image_cnt += 1
		if image_cnt>seq_len != 0:
			break
		ret, image = cap.read()
		if image is None:
			break

		if resolution != 0:
			image = make_image_square(image)
			image = cv2.resize(image, (resolution, resolution))
		image_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	cap.release()
	# im的形状是（256,256,3）
	if len(image_list)<seq_len:
		print(f'\nERROR: image_list is {seq_len}!!\n')
	im = np.stack(image_list, axis=3)
	# stack操作后 im的形状是（256,256,3,29）
	im = np.transpose(im, (3, 2, 0, 1))
	# im的形状是（29,3,256,256）
	im_tensor = torch.FloatTensor(im)

	return im_tensor

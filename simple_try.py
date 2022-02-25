import os
import glob
import shutil
import cv2
import numpy as np
import scipy.io.wavfile as wavf
from utils.GetDataFromFile import get_frame_and_wav_cv2
from utils.extract_wav import extract_wav

data_dir = '/home/tliu/fsx/dataset/class50/class-01'
mp4list = glob.glob(os.path.join(data_dir, '*.mp4'))

for idx, mp4name in enumerate(mp4list):
	if idx>1:
		break

	tgt_fps = 25.0
	extract_wav(mp4name)
	img_tensor, wav_tensor = get_frame_and_wav_cv2(mp4name, tgt_frame_num=100, tgt_fps=tgt_fps)
	img_array = img_tensor.numpy().astype(np.uint8)
	wav_array = wav_tensor.numpy()
	frame_num, channel, width, height = img_tensor.shape
	print(frame_num, channel, width, height)

	basename = os.path.basename(mp4name)
	mp4_name = os.path.join('/home/tliu/fsx/project/AVsync/tmp', basename)
	wav_name = mp4_name[:-3]+'wav'
	out_name = os.path.join('/home/tliu/fsx/project/AVsync/out', basename)
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	output_movie = cv2.VideoWriter(mp4_name, fourcc, tgt_fps, (height, width))
	for frame in img_array:
		frame = frame.transpose(1, 2, 0)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		output_movie.write(frame)
	output_movie.release()
	wavf.write(wav_name, 16000, wav_array)
	os.system(f'ffmpeg -y -i {mp4_name} -i {wav_name} -strict -2 -f mp4 {out_name} -loglevel quiet')

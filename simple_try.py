import os
import glob
import shutil
import cv2
from utils.GetDataFromFile import get_frame_and_wav_cv2
from utils.extract_wav import extract_wav

data_dir = '/home/tliu/fsx/dataset/class50/class-01'
mp4list = glob.glob(os.path.join(data_dir, '*.mp4'))

for idx, mp4name in enumerate(mp4list):
	if idx>20:
		break

	tgt_fps = 25
	extract_wav(mp4name)
	img_tensor, wav_tensor = get_frame_and_wav_cv2(mp4name, tgt_fps=tgt_fps)
	img_array = img_tensor.numpy().astype(int)
	wav_array = wav_tensor.numpy().astype(int)
	frame_num, channel, width, height = img_tensor.shape
	print(frame_num, channel, width, height)

	basename = os.path.basename(mp4name)
	output_name = os.path.join('/home/tliu/fsx/project/AVsync/tmp', basename)
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	output_movie = cv2.VideoWriter(output_name, fourcc, tgt_fps, (width, height))
	for frame in img_array:
		frame = frame.transpose(1, 2, 0)
		output_movie.write(frame)
	output_movie.release()

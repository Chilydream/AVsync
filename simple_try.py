import multiprocessing as mp
import os
import time
import shutil
import torch
import sys

import tqdm
import cv2


def main():
	root_dir = '/home/tliu/fsx/dataset/avspeech'
	flog = open('log/avspeech.log', 'w')
	file_list = os.listdir(root_dir)
	for filename in tqdm.tqdm(file_list):
		if filename.endswith('wav'):
			continue
		filename = os.path.join(root_dir, filename)
		pos0 = filename.rfind('_')
		pos1 = filename.rfind('-')
		pos2 = filename.rfind('.mp4')
		start_time = filename[pos0+1:pos1]
		end_time = filename[pos1+1:pos2]
		duration_real = float(end_time)-float(start_time)

		cap = cv2.VideoCapture(filename)
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps<10:
			print(f'{filename}\nfps:{fps}, duration:{duration_real}')
		print(f'{filename}\nfps:{fps}, duration:{duration_real}', file=flog)
		# h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		# w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		# n_cv2 = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		# n_real = 0
		# while True:
		# 	ret, image = cap.read()
		# 	if image is None:
		# 		break
		# 	n_real += 1
		# print(f'fps:{fps},n_cv2:{n_cv2},n_real:{n_real}')
		# print(f'duration_real:{duration_real}, duration_cv2:{n_cv2/fps}, duration_cnt:{n_real/fps}')
		cap.release()
	flog.close()


if __name__ == '__main__':
	main()

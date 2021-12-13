import glob
import os
import multiprocessing
import sys
import torch
import tqdm

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from third_party.HRNet.utils_inference import get_model_by_name
from utils.GetConsoleArgs import TrainOptions
from utils.extract_lmk import extract_lmk

args = TrainOptions('config/lab_sync.yaml').parse()
with open(args.train_list, 'w') as f_train,	open(args.val_list, 'w') as f_val, open(args.test_list, 'w') as f_test:
	dataset_dir = '/home/tliu/fsx/dataset/data1204'
	video_list = glob.glob(os.path.join(dataset_dir, '*'))
	for i, filename in enumerate(video_list):
		print(filename)
		if i%10<8:
			if filename[32]=='1':
				print(f'0\t{filename}', file=f_train)
			elif filename[32]=='2':
				print(f'1\t{filename}', file=f_train)
		elif i%10==8:
			if filename[32]=='1':
				print(f'0\t{filename}', file=f_val)
			elif filename[32]=='2':
				print(f'1\t{filename}', file=f_val)
		else:
			if filename[32]=='1':
				print(f'0\t{filename}', file=f_test)
			elif filename[32]=='2':
				print(f'1\t{filename}', file=f_test)

	dataset_dir = '/home/tliu/fsx/dataset/lab50-new/silent'
	video_list = glob.glob(os.path.join(dataset_dir, '*'))
	for i, filename in enumerate(video_list):
		if i%10<8:
			print(f'0\t{filename}', file=f_train)
		elif i%10==8:
			print(f'0\t{filename}', file=f_val)
		else:
			print(f'0\t{filename}', file=f_test)
	dataset_dir = '/home/tliu/fsx/dataset/lab50-new/talk'
	video_list = glob.glob(os.path.join(dataset_dir, '*'))
	for i, filename in enumerate(video_list):
		if i%10<8:
			print(f'1\t{filename}', file=f_train)
		elif i%10==8:
			print(f'1\t{filename}', file=f_val)
		else:
			print(f'1\t{filename}', file=f_test)

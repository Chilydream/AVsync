import glob
import os
import multiprocessing
import sys
import torch
import tqdm
from moviepy.editor import VideoFileClip

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from third_party.HRNet.utils_inference import get_model_by_name
from utils.GetConsoleArgs import TrainOptions
from utils.GetDataFromFile import get_frame_and_wav, get_frame_tensor
from utils.extract_lmk import extract_lmk

args = TrainOptions('config/lab_sync.yaml').parse()
with open(args.train_list, 'r') as fr:
	lines = fr.readlines()
	for line in lines:
		is_talk, filename = line.strip().split('\t')
		img_seq = get_frame_tensor(filename)
		print(img_seq.shape)
		break

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

args = TrainOptions('config/lmk2text.yaml').parse()
run_device = torch.device("cuda" if args.gpu else "cpu")
model_hrnet = get_model_by_name('300W', root_models_path='pretrain_model')
model_hrnet = model_hrnet.to(run_device).eval()

with open('metadata/LRW_train_3090.txt', 'r') as fr:
	lines = fr.readlines()
	for idx in range(args.lmk_thread, len(lines), 10):
		line = lines[idx]
		_, mp4name = line.strip().split('\t')
		extract_lmk(model_hrnet, mp4name, run_device)
		print(idx)

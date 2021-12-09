import math
import os
import platform
import time
import numpy as np
import cv2
import torch
import tqdm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import utils as vutils
import wandb
import glob
# import skvideo.io
# import imutils
# from moviepy.editor import VideoFileClip
import sys

from model.Lmk2LipModel import Lmk2LipModel
from utils.data_utils.LRWImageLmkTriplet import LRWImageLmkTripletDataLoader
from utils.extract_wav import extract_wav

# sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
# sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')
sys.path.append('./third_party/yolo')
sys.path.append('./third_party/HRNet')

from utils.GetDataFromFile import get_mfcc
from utils.extract_lmk import extract_lmk
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from utils.accuracy import topk_acc, get_gt_label, get_new_idx, get_rand_idx
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

args = TrainOptions('config/lmk2text.yaml').parse()
run_device = torch.device('cuda:0')
model_lmk2lip = Lmk2LipModel(lmk_emb=args.lmk_emb, lip_emb=args.lip_emb, stride=1)
model_lmk2lip.to(run_device)

with open('metadata/LRW_train_3090.txt', 'r') as fr, open('metadata/LRW_train_3090_lmk.txt', 'w') as fw:
	lines = fr.readlines()
	for line in lines:
		_, mp4name = line.strip().split('\t')
		lmkname = mp4name[:-3]+'lmk'
		lmk_tensor = torch.load(lmkname)
		if lmk_tensor.shape[0] != 29 or lmk_tensor.shape[1] != 68:
			print(f'Error file {mp4name}, lmk shape: {lmk_tensor.shape}')
			continue
		print(line.strip(), file=fw)

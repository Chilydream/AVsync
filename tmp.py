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

args = TrainOptions('config/speech2text.yaml').parse()

with open('metadata/test.txt', 'r') as fr:
	lines = fr.readlines()
	for line in lines:
		_, mp4name = line.strip().split('\t')
		wavname = mp4name[:-3]+'wav'
		if not os.path.exists(wavname):
			extract_wav(mp4name)

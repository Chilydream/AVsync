import math
import os
import platform
import time

import librosa
import numpy as np
import cv2
import torch
import torchaudio
import tqdm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import utils as vutils
import wandb
import glob
import skvideo.io
import imutils
from moviepy.editor import VideoFileClip
import sys

from model.Lmk2LipModel import Lmk2LipModel
from model.VGGModel import VGGVoice
from utils.data_utils.LRWImageLmkTriplet import LRWImageLmkTripletDataLoader
from utils.data_utils.LabRaw import LabDataset, LabDataLoader
from utils.extract_wav import extract_wav

sys.path.append('./third_party/yolo')
sys.path.append('./third_party/HRNet')

from utils.GetDataFromFile import get_mfcc, get_wav, get_frame_moviepy
from utils.extract_lmk import extract_lmk
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from utils.accuracy import topk_acc, get_gt_label, get_new_idx, get_rand_idx
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

args = TrainOptions('config/lab_sync.yaml').parse()
run_device = torch.device('cuda:0')
img_resolution = 256
face_resolution = 128

# model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
# model_yolo.to(run_device)
# model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt',
#                                       map_location=run_device))
# model_hrnet = get_model_by_name('300W', root_models_path='pretrain_model')
# model_hrnet = model_hrnet.to(run_device).eval()

train_loader = LabDataLoader(args.train_list, args.batch_size,
                             num_workers=args.num_workers,
                             n_mfcc=args.n_mfcc,
                             seq_len=args.seq_len,
                             resolution=args.resolution,
                             is_train=False, max_size=0)

for data in train_loader:
	a_wav, a_img, a_label = data
	print(a_wav.shape)
	print(a_img.shape)
	print(a_label)
	break

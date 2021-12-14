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
from utils.data_utils.LabLmkWav import LabLmkWavDataLoader
from utils.data_utils.LabRaw import LabDataset, LabDataLoader
from utils.extract_wav import extract_wav

sys.path.append('./third_party/yolo')
sys.path.append('./third_party/HRNet')

from utils.GetDataFromFile import get_mfcc, get_wav, get_frame_moviepy, get_frame_and_wav
from utils.extract_lmk import extract_lmk
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from utils.accuracy import topk_acc, get_gt_label, get_rand_idx
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

args = TrainOptions('config/lab_sync.yaml').parse()

a = torch.load('/home/tliu/fsx/dataset/data1204/2-p118.lmk')
# a = torch.load('/home/tliu/fsx/dataset/LRW/ABOUT/test/ABOUT_00002.lmk')
avg_lmk = torch.mean(a, dim=(0, 1))
std_lmk = torch.std(a, dim=(0, 1))
# scale_lmk0 = torch.max(a[:, :, 0]) - torch.min(a[:, :, 0])
# scale_lmk1 = torch.max(a[:, :, 1]) - torch.min(a[:, :, 1])
print(avg_lmk)
print(std_lmk)

a = (a-avg_lmk)/std_lmk
# a[..., 0] /= scale_lmk0
# a[..., 1] /= scale_lmk1
# print(a)
avg_lmk = torch.mean(a, dim=(0, 1))
# scale_lmk0 = torch.max(a[:, :, 0]) - torch.min(a[:, :, 0])
# scale_lmk1 = torch.max(a[:, :, 1]) - torch.min(a[:, :, 1])
std_lmk = torch.std(a, dim=(0, 1))
print(avg_lmk)
print(std_lmk)
# print(scale_lmk0)
# print(scale_lmk1)

import argparse
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
import yaml
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
from model.VGGModel import VGG6_speech
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

args = TrainOptions('config/sync_multisensory.yaml').parse()

src_fps = 25
tgt_fps = 30
raw_frame_num = 25
frac_ratio = src_fps/tgt_fps
new_idx = list(map(lambda i: int(i*frac_ratio), range(math.ceil(raw_frame_num/frac_ratio))))
print(new_idx)
print(len(new_idx))

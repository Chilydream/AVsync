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
from utils.extract_wav import extract_wav

sys.path.append('./third_party/yolo')
sys.path.append('./third_party/HRNet')

from utils.GetDataFromFile import get_mfcc, get_wav
from utils.extract_lmk import extract_lmk
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from utils.accuracy import topk_acc, get_gt_label, get_new_idx, get_rand_idx
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

mp4name = 'test/2cut2.mp4'
wavname = mp4name[:-3]+'wav'
wav_array = get_wav(wavname)
wav_tensor = torch.tensor(wav_array)
# torch.Size([batch_size, 19456])
torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                               hop_length=175, f_min=0.0, f_max=8000,
                                               pad=0, n_mels=40)
mfcc_tensor = torchfb(wav_tensor)
# torch.Size([batch_size, nmfcc=40, 112])

video_file_clip = VideoFileClip(mp4name)
video_file_clip = video_file_clip.to_RGB()
f0 = video_file_clip.make_frame(0)
print(type(f0))
print(f0.shape)
# audio_file_clip = video_file_clip.audio
# a = audio_file_clip.to_soundarray(fps=16000)
# a = a[:, 0]
# a = torch.tensor(a, dtype=torch.float32)
# print(a.shape)
# b = torchfb(a)
# print(b.shape)
video_file_clip.close()

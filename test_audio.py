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

args = TrainOptions('config/speech2text.yaml').parse()

model_wav2v = VGG6_speech()
torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                               hop_length=160, f_min=0.0, f_max=8000,
                                               pad=0, n_mels=40)

# mp4name = 'test/ABOUT_00001.mp4'
mp4name = 'test/avspeech/elcpPYx4X2c_150.000000-156.280000.mp4'
# mp4name = 'test/avspeech/es-nUdEm8sQ_120.019000-125.392000.mp4'
extract_wav(mp4name)
wavname = mp4name[:-3]+'wav'
y, sr = librosa.load(wavname, sr=16000)
print(y.shape, sr)
y = torch.FloatTensor(y).unsqueeze(0)
a_voice = model_wav2v(y)

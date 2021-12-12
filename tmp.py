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

from utils.GetDataFromFile import get_mfcc, get_wav, get_frame_moviepy
from utils.extract_lmk import extract_lmk
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from utils.accuracy import topk_acc, get_gt_label, get_new_idx, get_rand_idx
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

run_device = torch.device('cuda:0')
img_resolution = 256
face_resolution = 128

# model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
# model_yolo.to(run_device)
# model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt',
#                                       map_location=run_device))
# model_hrnet = get_model_by_name('300W', root_models_path='pretrain_model')
# model_hrnet = model_hrnet.to(run_device).eval()

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

# pad_resize = transforms.Compose([PadSquare(),
#                                  transforms.Resize((face_resolution, face_resolution))])
# img_tensor = get_frame_moviepy(mp4name, resolution=img_resolution)
# img_tensor = img_tensor.to(run_device)
# print(img_tensor.shape)
#
# with torch.no_grad():
#     bbox_list = face_detect(model_yolo, img_tensor)
#     face_list = []
#     for i in range(len(bbox_list)):
#         x1, y1, x2, y2 = bbox_list[i]
#         if x1>=x2:
#             x1, x2 = 0, img_resolution-1
#         if y1>=y2:
#             y1, y2 = 0, img_resolution-1
#         crop_img = img_tensor[i, :, y1:y2, x1:x2]
#         face_list.append(pad_resize(crop_img))
#     face_tensor = torch.stack(face_list, dim=0).to(run_device)
#     print(face_tensor.shape)
#     lmk_list = get_batch_lmks(model_hrnet, face_tensor, output_size=(face_resolution, face_resolution))
#     print(lmk_list.shape)
#     print(type(lmk_list))

video_file_clip = VideoFileClip(mp4name)
audio_file_clip = video_file_clip.audio
a = audio_file_clip.to_soundarray(fps=16000)[:, 0]
a = torch.tensor(a, dtype=torch.float32)
print(a.shape)
b = torchfb(a)
print(b.shape)
video_file_clip.close()

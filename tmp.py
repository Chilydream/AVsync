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
from model.VGGModel import VGGVoice
from utils.data_utils.LRWImageLmkTriplet import LRWImageLmkTripletDataLoader
from utils.extract_wav import extract_wav

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

args = TrainOptions('config/train.yaml').parse()
run_device = torch.device('cuda:0')
model_lmk2lip = Lmk2LipModel(lmk_emb=args.lmk_emb, lip_emb=args.lip_emb, stride=1)
model_wav2v = VGGVoice(n_out=args.voice_emb)
l2t_ckpt = torch.load('pretrain_model/lmk2t.model')
model_lmk2lip.load_state_dict(l2t_ckpt['model_lmk2lip'])
s2t_ckpt = torch.load('pretrain_model/s2t.model')
model_wav2v.load_state_dict(s2t_ckpt['model_wav2v'])
save_dict = {
	'model_lmk2lip': model_lmk2lip.state_dict(),
	'model_wav2v': model_wav2v.state_dict(),
}
torch.save(save_dict, 'pretrain_model/pre_av.model')

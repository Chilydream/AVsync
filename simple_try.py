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

a = torch.rand((4, 3, 2))
print(a.shape)
b = torch.mean(a, dim=(0, 1))
print(b.shape)

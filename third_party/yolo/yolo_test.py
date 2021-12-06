import os
import time
from typing import OrderedDict

import numpy as np
import torch
import pymongo
import cv2
import shutil
import sys
import torch.nn as nn

sys.path.append('/root/ChineseDataset/AVsync/third_party/yolo')
from yolo_utils.util_yolo import face_detect
from yolo_models.yolo import Model as yolo_model

run_device = torch.device('cuda:0')

a = yolo_model(cfg='/root/ChineseDataset/AVsync/config/yolov5s.yaml').float().fuse().eval()
a.to(run_device)
a.load_state_dict(torch.load('/root/ChineseDataset/AVsync/pretrain_model/raw_yolov5s.pt',
                             map_location=run_device))


this_video = cv2.VideoCapture('/root/ChineseDataset/AVsync/test/CHINA_00039.mp4')
_, frame = this_video.read()
cv2.imwrite('/root/ChineseDataset/AVsync/test/CHINA.jpg', frame)
bbox_a = face_detect(a, frame)
print(bbox_a)


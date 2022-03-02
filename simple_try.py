import os
import glob
import shutil
import cv2
import numpy as np
import scipy.io.wavfile as wavf
from utils.GetDataFromFile import get_frame_and_wav_cv2
from utils.extract_wav import extract_wav

dataset_dir = '/home/tliu/fsx/dataset/LRS2'
main_dir = os.path.join(dataset_dir, 'main')
pretrain_dir = os.path.join(dataset_dir, 'pretrain')
video_list = glob.glob(os.path.join(main_dir, '*', '*.mp4'))
print(len(video_list))
video_list.extend(glob.glob(os.path.join(pretrain_dir, '*', '*.mp4')))
print(len(video_list))

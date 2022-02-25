import os
import glob
import shutil
import cv2
from utils.GetDataFromFile import get_frame_and_wav_cv2
from utils.extract_wav import extract_wav

data_dir = '/home/tliu/fsx/dataset/class50/class-01'
mp4list = glob.glob(os.path.join(data_dir, '*.mp4'))

a = cv2.imread('/home/tliu/fsx/project/AVsync/test/bai.jpg')
print(a.shape)
print(a.dtype)

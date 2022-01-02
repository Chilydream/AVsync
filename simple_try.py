import multiprocessing as mp
import os
import time
import shutil
import torch
import sys

import tqdm
import cv2

sys.path.append('./third_party/yolo')
from utils.GetDataFromFile import get_frame_tensor
from utils.crop_face import crop_face_seq
from third_party.yolo.yolo_models.yolo import Model as yolo_model


def main():
	ftrain = open('metadata/avspeech_train.txt', 'r')
	lines = ftrain.readlines()
	for line in lines:
		filename = line.strip()
		cap = cv2.VideoCapture(filename)
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps<15:
			print(fps, filename)
		cap.release()


if __name__ == '__main__':
	main()

import multiprocessing as mp
import os
import time
import shutil
import torch
import sys

import tqdm

sys.path.append('./third_party/yolo')
from utils.GetDataFromFile import get_frame_tensor
from utils.crop_face import crop_face_seq
from third_party.yolo.yolo_models.yolo import Model as yolo_model


def main():
	with open('metadata/avspeech_raw.txt', 'r') as fr:
		lines = fr.readlines()
		for line in tqdm.tqdm(lines):
			filename = '/media/tliu/AVSpeech/out/AVSpeech/'+line.strip()
			newname = '/home/tliu/fsx/dataset/avspeech/'+line.strip()
			shutil.copy(filename, newname)


if __name__ == '__main__':
	main()

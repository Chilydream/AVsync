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
	ftrain = open('metadata/avspeech_train.txt', 'w')
	fval = open('metadata/avspeech_val.txt', 'w')
	ftest = open('metadata/avspeech_test.txt', 'w')
	with open('metadata/avspeech_raw.txt', 'r') as fr:
		lines = fr.readlines()
		for idx, line in enumerate(lines):
			newname = '/home/tliu/fsx/dataset/avspeech/'+line.strip()
			if idx%10<8:
				print(f'{newname}', file=ftrain)
			elif idx%10==8:
				print(f'{newname}', file=fval)
			else:
				print(f'{newname}', file=ftest)
	ftrain.close()
	fval.close()
	ftest.close()


if __name__ == '__main__':
	main()

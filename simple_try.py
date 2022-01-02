import multiprocessing as mp
import os
import time
import shutil
import torch
import sys

import tqdm

sys.path.append('./third_party/yolo')
from utils.GetDataFromFile import get_frame_tensor, get_frame_and_wav
from utils.crop_face import crop_face_seq
from third_party.yolo.yolo_models.yolo import Model as yolo_model


def main():
	a = torch.rand((32, 256, 19, 1))
	print(a.shape)
	b = list(range(1, 32))
	b.append(0)
	print(b)
	c = a[b, ...]
	print(c.shape)


if __name__ == '__main__':
	main()

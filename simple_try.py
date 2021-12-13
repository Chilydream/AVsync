import glob
import os
import multiprocessing
import sys
import torch
import tqdm
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from utils.GetDataFromFile import get_frame_tensor
from model.Lmk2LipModel import Lmk2LipModel
from model.VGGModel import VGGVoice
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks

args = TrainOptions('config/lab_sync.yaml').parse()
run_device = torch.device('cuda:0')

pad_resize = transforms.Compose([PadSquare(),
                                 transforms.Resize(args.face_resolution)])

model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt'))
model_yolo.to(run_device)
model_hrnet = get_model_by_name('300W', root_models_path='pretrain_model')
model_hrnet = model_hrnet.eval()
model_hrnet.to(run_device)

with open(args.train_list, 'r') as fr:
	lines = fr.readlines()
	cnt = 0
	for line in lines:
		is_talk, filename = line.strip().split('\t')
		lmkname = filename[:-3]+'lmk'
		cnt += 1
		if os.path.exists(lmkname):
			continue
		print(f'processing {cnt:03d}/{len(lines)} file {filename}')
		img_seq = get_frame_tensor(filename, resolution=256, seq_len=500)
		img_seq = img_seq.to(run_device)
		print(img_seq.shape)
		bbox_list = face_detect(model_yolo, img_seq)
		face_list = []
		for i, bbox in enumerate(bbox_list):
			x1, y1, x2, y2 = bbox
			if x1>=x2:
				x1, x2 = 0, args.img_resolution-1
			if y1>=y2:
				y1, y2 = 0, args.img_resolution-1
			face_list.append(pad_resize(img_seq[i, :, y1:y2, x1:x2]))
		face_tensor = torch.stack(face_list).to(run_device)
		lmk_seq = get_batch_lmks(model_hrnet, face_tensor,
		                         output_size=(args.face_resolution, args.face_resolution))
		torch.save(lmk_seq, lmkname)
		del face_list, lmk_seq, img_seq, face_tensor
		torch.cuda.empty_cache()

import math
import os
import platform
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
import sys

if platform.system() == "Windows":
	sys.path.append('D:/project/AVsync/third_party/yolo')
else:
	sys.path.append('/root/ChineseDataset/AVsync/third_party/yolo')

from model.FaceModel import FaceModel
from model.SpeakModel import SpeakModel
from utils.data_utils.LRWRaw import LRWDataLoader
from utils.data_utils.LabRaw import LabDataLoader
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from utils.tensor_utils import PadSquare
from utils.accuracy import topk_acc, get_new_idx, get_gt_label, get_rand_idx
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/speak.yaml').parse()
	start_epoch = 0
	batch_size = args.batch_size
	batch_first = True
	torch.backends.cudnn.benchmark = args.gpu
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	cur_exp_path = os.path.join(args.exp_dir, args.exp_num)
	cache_dir = os.path.join(cur_exp_path, 'cache')
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	path_train_log = os.path.join(cur_exp_path, 'train.log')

	# ============================WandB日志=============================
	if args.wandb:
		wandb.init(project='Speak Silent', config=args)

	# ============================模型载入===============================
	print('%sStart loading model%s'%('='*20, '='*20))
	model_face = FaceModel(n_out=args.face_eb)
	model_speak = SpeakModel(face_emb=args.face_eb, hid_emb=128, batch_first=batch_first)
	model_face.to(run_device)
	model_speak.to(run_device)

	optim_face = optim.SGD(model_face.parameters(), lr=args.face_lr, momentum=0.5)
	optim_speak = optim.Adam(model_speak.parameters(), lr=args.speak_lr, betas=(0.9, 0.999))
	criterion = nn.L1Loss()
	model_face.train()
	model_speak.train()
	sch_face = optim.lr_scheduler.ExponentialLR(optim_face, gamma=args.face_gamma)
	sch_speak = optim.lr_scheduler.ExponentialLR(optim_speak, gamma=args.speak_gamma)

	model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
	model_yolo.to(run_device)
	model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt',
	                                      map_location=run_device))

	pad_resize = transforms.Compose([PadSquare(),
	                                 transforms.Resize((args.resolution, args.resolution))])
	if args.wandb:
		wandb.watch(model_face)
		wandb.watch(model_speak)

	# ============================度量载入===============================
	epoch_loss_ss = Meter('Speak Silent Loss', 'avg', ':.2f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.2f')
	epoch_acc_ss = Meter('Speak Silent ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':3.0f')

	epoch_reset_list = [epoch_loss_final, epoch_timer, epoch_acc_ss]
	print('Train Parameters')
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}')
	print('')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('%sStart loading dataset%s'%('='*20, '='*20))
	loader_timer.set_start_time(time.time())
	train_loader = LabDataLoader(args.train_list, batch_size, args.num_workers,
	                             seq_len=16, is_train=True, max_size=50000)
	# valid_loader = LRWDataLoader(args.valid_list, batch_size, args.num_workers,
	#                             args.n_mfcc, is_train=False)
	loader_timer.update(time.time())
	print('Finish loading dataset', loader_timer)

	# ========================预加载模型================================
	if args.mode.lower() in ['test', 'valid', 'eval']:
		model_ckpt = torch.load(args.pretrain_model)
		model_face.load_state_dict(model_ckpt['model_face'])
		model_speak.load_state_dict(model_ckpt['model_speak'])
		# todo: 还没有编写测试代码
		return
	elif args.mode.lower().lower() in ['continue']:
		print('Loading pretrained model', args.pretrain_model)
		model_ckpt = torch.load(args.pretrain_model)
		model_face.load_state_dict(model_ckpt['model_face'])
		model_speak.load_state_dict(model_ckpt['model_speak'])
		optim_face.load_state_dict(model_ckpt['optim_face'])
		optim_speak.load_state_dict(model_ckpt['optim_speak'])
		start_epoch = model_ckpt['epoch']
		file_train_log = open(path_train_log, 'a')
	elif args.mode.lower() in ['train']:
		file_train_log = open(path_train_log, 'w')
	else:
		raise Exception(f"未知训练模式{args.mode}")

	print('Train Parameters', file=file_train_log)
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}', file=file_train_log)
	print('', file=file_train_log)

	# ============================开始训练===============================
	print('%sStart Training%s'%('='*20, '='*20))
	for epoch in range(start_epoch, args.epoch):
		print('\nEpoch: %d'%epoch)
		batch_cnt = 0
		epoch_timer.set_start_time(time.time())
		for data in train_loader:
			# image_data = (batch,29,3,256,256)
			speak_label = data[1]
			image_data = data[0].to(run_device)
			image_data.transpose_(1, 0)
			# image_data = (29,batch,3,256,256)
			face_emb_list = []
			for i, image_batch in enumerate(image_data):
				with torch.no_grad():
					bbox_seq = face_detect(model_yolo, image_batch)
				for j, bbox in enumerate(bbox_seq):
					x1, y1, x2, y2 = bbox_seq[j]
					if y1<=y2:
						y1, y2 = 0, 255
					if x1<=x2:
						x1, x2 = 0, 255
					crop_face = image_batch[j, :, y1:y2, x1:x2]
					image_data[i, j] = pad_resize(crop_face)
				# todo: 考虑在这里保存一下预处理后的数据，loader额外返回文件名
				face_batch = model_face(image_batch)
				face_emb_list.append(face_batch)
			face_data = torch.stack(face_emb_list, dim=0)
			# face_data = (seq_len, batch, face_emb)

			label_gt = speak_label.float().squeeze().to(run_device)
			if batch_first:
				face_data.transpose_(1, 0)
			label_pred = model_speak(face_data)

			# ======================计算沉默说话损失===========================
			loss_avmatch = criterion(label_gt, label_pred)
			batch_acc_avmatch = torch.sum(label_gt*label_pred>0).item()

			# ==========================反向传播===============================
			optim_face.zero_grad()
			optim_speak.zero_grad()
			batch_loss_final = loss_avmatch
			batch_loss_final.backward()
			optim_face.step()
			optim_speak.step()

			# ==========计量更新============================
			epoch_acc_ss.update(batch_acc_avmatch*100/batch_size)
			epoch_loss_ss.update(loss_avmatch.item())
			epoch_loss_final.update(batch_loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:03d}/{len(train_loader):03d}  {epoch_timer}{epoch_loss_final}{epoch_acc_ss}',
			      end='     ')

		sch_face.step()
		sch_speak.step()
		print('')
		print('Epoch:', epoch, epoch_loss_final, epoch_acc_ss,
		      file=file_train_log)
		log_dict = {'final loss': epoch_loss_final.avg,
		            'ss loss': epoch_loss_ss.avg,
		            'ss acc': epoch_acc_ss.avg}
		for meter in epoch_reset_list:
			meter.reset()

		# =======================保存模型=======================
		if args.gpu:
			torch.cuda.empty_cache()

		if (epoch+1)%args.save_step == 0:
			torch.save({'epoch': epoch+1,
			            'model_face': model_face.state_dict(),
			            'model_speak': model_speak.state_dict(),
			            'optim_face': optim_face.state_dict(),
			            'optim_speak': optim_speak.state_dict(),
			            'sch_face': sch_face.state_dict(),
			            'sch_speak': sch_speak.state_dict(), },
			           cache_dir+"/model%09d.model"%epoch)

		# ===========================验证=======================
		valid_log = dict()
		if (epoch+1)%args.valid_step == 0:
			model_face.eval()
			model_speak.eval()
			criterion.eval()
			# todo: 测试代码
			model_face.train()
			model_speak.train()
			criterion.train()
		log_dict.update(valid_log)

		if args.wandb:
			wandb.log(log_dict)
	file_train_log.close()
	if args.wandb:
		wandb.finish()


if __name__ == '__main__':
	main()

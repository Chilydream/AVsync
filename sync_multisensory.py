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
from pytorch_metric_learning import losses
import sys

from model.MultiSensory import MultiSensory
from utils.data_utils.LabRaw import LabDataLoader

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from utils.crop_face import crop_face_batch_seq
from utils.accuracy import get_gt_label, get_rand_idx
from utils.data_utils.LRWRaw import LRWDataLoader
from model.Lip2TModel import Lip2T_fc_Model
from model.Lmk2LipModel import Lmk2LipModel
from model.SyncModel import SyncModel
from model.Voice2TModel import Voice2T_fc_Model
from model.VGGModel import VGG6_speech, ResLip, VGG6_lip, VGG5_lip
from utils.data_utils.LRWImageTriplet import LRWImageTripletDataLoader
from utils.tensor_utils import PadSquare, MyContrastiveLoss
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks
from third_party.yolo.yolo_models.yolo import Model as yolo_model


def lab_run(model_ms, data, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")
	a_img, a_wav_match, a_wav_mis = data
	a_img = a_img.to(run_device)
	a_img.transpose_(2, 1)
	a_wav_match = a_wav_match.to(run_device)
	a_wav_mis = a_wav_mis.to(run_device)
	label_gt_mis = torch.zeros(args.batch_size, dtype=torch.long)
	label_gt_match = torch.ones(args.batch_size, dtype=torch.long)
	a_lip = model_ms.img_forward(a_img)
	a_snd_match = model_ms.snd_forward(a_wav_match)
	a_snd_mis = model_ms.snd_forward(a_wav_mis)
	label_pred_match = model_ms.merge_forward(snd_feature=a_snd_match, img_feature=a_lip)
	label_pred_mis = model_ms.merge_forward(snd_feature=a_snd_mis, img_feature=a_lip)

	label_gt = torch.cat((label_gt_match, label_gt_mis), dim=0)
	label_pred = torch.cat((label_pred_match, label_pred_mis), dim=0)
	return label_gt, label_pred


def lrw_run(model_ms, data, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")
	a_wav, a_img, a_wid = data
	a_wav = a_wav.to(run_device)
	a_img = a_img.to(run_device)
	a_wid = a_wid.to(run_device)
	# a_face = crop_face_batch_seq(model_yolo, a_img, args)
	a_img.transpose_(2, 1)
	a_lip = model_ms.img_forwad(a_img)

	new_idx = get_rand_idx(args.batch_size)
	a_wav = a_wav[new_idx, :]
	a_voice = model_ms.snd_forward(a_wav)

	label_gt = get_gt_label(a_wid, new_idx).to(run_device)
	label_pred = model_ms.merge_forward(img_feature=a_lip, snd_feature=a_voice)
	return label_gt, label_pred


def evaluate(model_ms, criterion_class, loader, args):
	with torch.no_grad():
		model_ms.eval()
		criterion_class.eval()

		val_loss_class = Meter('Class Loss', 'avg', ':.4f')
		val_loss_final = Meter('Final Loss', 'avg', ':.4f')
		val_acc_class = Meter('Class ACC', 'avg', ':.2f', '%,')
		val_timer = Meter('Time', 'time', ':3.0f')
		val_timer.set_start_time(time.time())

		print('\tEvaluating Result:')
		for data in loader:
			label_gt, label_pred = lrw_run(model_ms, data, args)

			# ======================计算唇部特征单词分类损失===========================
			loss_class = criterion_class(label_pred, label_gt)
			correct_num_class = torch.sum(torch.argmax(label_pred, dim=1) == label_gt).item()

			loss_final = loss_class

			# ==========计量更新============================
			val_acc_class.update(correct_num_class*100/len(label_gt))
			val_loss_class.update(loss_class.item())
			val_loss_final.update(loss_final.item())
			val_timer.update(time.time())
			print(f'\r\tBatch:{val_timer.count:04d}/{len(loader):04d}  {val_timer}{val_loss_final}',
			      f'{val_acc_class} EMA ACC: {val_acc_class.avg_ema:.2f}%, ',
			      f'{val_loss_class}',
			      sep='', end='     ')

		model_ms.train()
		criterion_class.train()
		val_log = {'val_loss_class': val_loss_class.avg,
		           'val_loss_final': val_loss_final.avg,
		           'val_acc_class': val_acc_class.avg}
		return val_log


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/sync_multisensory.yaml').parse()
	start_epoch = 0
	batch_size = args.batch_size
	torch.backends.cudnn.benchmark = args.gpu
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	cur_exp_path = os.path.join(args.exp_dir, args.exp_num)
	cache_dir = os.path.join(cur_exp_path, 'cache')
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	path_train_log = os.path.join(cur_exp_path, 'train.log')

	# ============================WandB日志=============================
	if args.wandb:
		wandb.init(project=args.project_name, config=args,
		           name=args.exp_num, group=args.exp_num)

	# ============================模型载入===============================
	print('%sStart loading model%s'%('='*20, '='*20))

	model_ms = MultiSensory(sound_rate=16000, image_fps=25)
	model_list = [model_ms]
	for model_iter in model_list:
		model_iter.to(run_device)
		model_iter.train()

	optim_ms = optim.Adam(model_ms.parameters(), lr=args.sync_lr, betas=(0.9, 0.999))
	criterion_class = nn.CrossEntropyLoss()
	sch_ms = optim.lr_scheduler.ExponentialLR(optim_ms, gamma=args.sync_gamma)
	tosave_list = [
		'model_ms',
		'optim_ms',
		'sch_ms',
	]
	if args.wandb:
		for model_iter in model_list:
			wandb.watch(model_iter)

	# ============================度量载入===============================
	epoch_loss_class = Meter('Class Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.4f')
	epoch_acc_sync = Meter('Class ACC', 'avg', ':.2f', '%, ')
	epoch_timer = Meter('Time', 'time', ':4.0f')

	epoch_reset_list = [
		epoch_loss_final, epoch_loss_class,
		epoch_acc_sync,
		epoch_timer,
	]
	if args.mode in ('train', 'continue'):
		print('Train Parameters')
		for key, value in args.__dict__.items():
			print(f'{key:18}:\t{value}')
		print('')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('%sStart loading dataset%s'%('='*20, '='*20))
	loader_timer.set_start_time(time.time())
	# train_loader = LRWDataLoader(args.train_list, batch_size,
	#                              num_workers=args.num_workers,
	#                              n_mfcc=args.n_mfcc,
	#                              resolution=args.img_size,
	#                              seq_len=args.seq_len,
	#                              is_train=True, max_size=0)
	#
	# valid_loader = LRWDataLoader(args.val_list, batch_size,
	#                              num_workers=args.num_workers,
	#                              n_mfcc=args.n_mfcc,
	#                              resolution=args.img_size,
	#                              seq_len=args.seq_len,
	#                              is_train=True, max_size=0)
	train_loader = LabDataLoader(args.train_list, batch_size,
	                             num_workers=args.num_workers,
	                             seq_len=args.seq_len,
	                             resolution=args.img_size,
	                             is_train=True, max_size=0)
	valid_loader = LabDataLoader(args.val_list, batch_size,
	                             num_workers=args.num_workers,
	                             seq_len=args.seq_len,
	                             resolution=args.img_size,
	                             is_train=False, max_size=0)
	loader_timer.update(time.time())
	print(f'Batch Num in Train Loader: {len(train_loader)}')
	print(f'Finish loading dataset {loader_timer}')

	# ========================预加载模型================================
	if args.mode.lower() in ['test', 'valid', 'eval', 'val', 'evaluate']:
		del train_loader
		model_ckpt = torch.load(args.pretrain_model)
		for item_str in tosave_list:
			item_model = locals()[item_str]
			item_model.load_state_dict(model_ckpt[item_str])
		del model_ckpt
		print(f'\n{"="*20}Start Evaluating{"="*20}')
		if args.mode.lower() in ['valid', 'val', 'eval', 'evaluate']:
			evaluate(model_ms,
			         criterion_class,
			         valid_loader, args)
		else:
			del valid_loader
			test_loader = LRWDataLoader(args.test_list, batch_size,
			                            num_workers=args.num_workers,
			                            n_mfcc=args.n_mfcc,
			                            resolution=args.img_size,
			                            seq_len=args.seq_len,
			                            is_train=False, max_size=0)
			evaluate(model_ms,
			         criterion_class,
			         test_loader, args)
		print(f'\n\nFinish Evaluation\n\n')
		return
	elif args.mode.lower() in ['continue']:
		print('Loading pretrained model', args.pretrain_model)
		model_ckpt = torch.load(args.pretrain_model)
		for item_str in tosave_list:
			if item_str in model_ckpt.keys():
				item_model = locals()[item_str]
				item_model.load_state_dict(model_ckpt[item_str])
		start_epoch = model_ckpt['epoch']
		file_train_log = open(path_train_log, 'a')
	elif args.mode.lower() in ['train']:
		file_train_log = open(path_train_log, 'w')
		if args.pretrain_model is not None:
			model_ckpt = torch.load(args.pretrain_model)
			for item_str in tosave_list:
				if item_str in model_ckpt.keys():
					item_model = locals()[item_str]
					item_model.load_state_dict(model_ckpt[item_str])
	else:
		raise Exception(f"未定义训练模式{args.mode}")

	print('Train Parameters', file=file_train_log)
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}', file=file_train_log)
	print('\n', file=file_train_log)

	# ============================开始训练===============================
	print('%sStart Training%s'%('='*20, '='*20))
	for epoch in range(start_epoch, args.epoch):
		print('\nEpoch: %d'%epoch)
		batch_cnt = 0
		epoch_timer.set_start_time(time.time())
		for data in train_loader:
			# label_gt, label_pred = lrw_run(model_ms, data, args)
			label_gt, label_pred = lab_run(model_ms, data, args)

			# ======================计算唇部特征单词分类损失===========================
			loss_class = criterion_class(label_pred, label_gt)
			correct_num_class = torch.sum(torch.argmax(label_pred, dim=1) == label_gt).item()

			# ==========================反向传播===============================
			optim_ms.zero_grad()
			loss_final = loss_class
			loss_final.backward()
			optim_ms.step()

			# ==========计量更新============================
			epoch_acc_sync.update(correct_num_class*100/len(label_gt))
			epoch_loss_class.update(loss_class.item())
			epoch_loss_final.update(loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:04d}/{len(train_loader):04d}  {epoch_timer}{epoch_loss_final}',
			      f'{epoch_acc_sync}EMA ACC: {epoch_acc_sync.avg_ema:.2f}%, ',
			      f'{epoch_loss_class}',
			      sep='', end='     ')

		sch_ms.step()
		print('')
		print(f'Current Model M2V Learning Rate is {sch_ms.get_last_lr()}')
		log_dict = {'epoch': epoch,
		            epoch_loss_final.name: epoch_loss_final.avg,
		            epoch_loss_class.name: epoch_loss_class.avg,
		            epoch_acc_sync.name: epoch_acc_sync.avg}
		for meter in epoch_reset_list:
			meter.reset()

		# =======================保存模型=======================
		if args.gpu:
			torch.cuda.empty_cache()

		if (epoch+1)%args.save_step == 0:
			ckpt_dict = {'epoch': epoch+1}
			for item_str in tosave_list:
				item_model = locals()[item_str]
				ckpt_dict.update({item_str: item_model.state_dict()})
			print(f'save model to {cache_dir+"/model%09d.model"%epoch}')
			torch.save(ckpt_dict, cache_dir+"/model%09d.model"%epoch)

		# ===========================验证=======================
		if args.valid_step>0 and (epoch+1)%args.valid_step == 0:
			try:
				log_dict.update(evaluate(model_ms,criterion_class, valid_loader, args))
			except:
				print('Evaluating Error')

		if args.wandb:
			wandb.log(log_dict)
		print(log_dict, file=file_train_log)
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
	file_train_log.close()
	if args.wandb:
		wandb.finish()


if __name__ == '__main__':
	# with torch.autograd.set_detect_anomaly(True):
	main()

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

from model.SyncModel import SyncModel2
from utils.accuracy import get_new_idx, get_gt_label, get_rand_idx
from utils.data_utils.LRWImageLmkTriplet import LRWImageLmkTripletDataLoader
from utils.data_utils.LRWRaw import LRWDataLoader
from utils.data_utils.LRWTriplet import LRWTripletDataLoader

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from model.Lip2TModel import Lip2T_fc_Model
from model.Lmk2LipModel import Lmk2LipModel
from model.Voice2TModel import Voice2T_fc_Model
from model.VGGModel import VGGVoice, ResLip, VGGLip
from utils.data_utils.LRWImageTriplet import LRWImageTripletDataLoader
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks


def evaluate(model_lmk2lip, model_wav2v, model_sync, criterion_class, loader, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	val_loss_class = Meter('Class Loss', 'avg', ':.4f')
	val_loss_final = Meter('Final Loss', 'avg', ':.4f')
	val_acc_class = Meter('Class ACC', 'avg', ':.2f', '%,')
	val_timer = Meter('Time', 'time', ':3.0f')
	val_timer.set_start_time(time.time())

	print('\tEvaluating Result:')
	for data in loader:
		a_wav, a_lmk, a_wid = data
		a_lip = model_lmk2lip(a_lmk)
		a_voice = model_wav2v(a_wav)

		new_idx = get_new_idx(args.batch_size)
		a_voice = a_voice[:, new_idx, :]
		label_gt = get_gt_label(a_wid, new_idx).to(run_device)
		label_pred = model_sync(a_lip, a_voice)

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

	val_log = {'val_loss_class': val_loss_class.avg,
	           'val_loss_final': val_loss_final.avg,
	           'val_acc_class': val_acc_class.avg}
	return val_log


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/train.yaml').parse()
	start_epoch = 0
	batch_size = args.batch_size
	batch_first = args.batch_first
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

	model_lmk2lip = Lmk2LipModel(lmk_emb=args.lmk_emb, lip_emb=args.lip_emb, stride=1)
	model_wav2v = VGGVoice(n_out=args.voice_emb)
	model_sync = SyncModel2(lip_emb=args.lip_emb, voice_emb=args.voice_emb)
	model_list = [model_lmk2lip, model_wav2v, model_sync]
	for model_iter in model_list:
		model_iter.to(run_device)
		model_iter.train()

	optim_lmk2lip = optim.Adam(model_lmk2lip.parameters(), lr=args.lmk2lip_lr, betas=(0.9, 0.999))
	optim_wav2v = optim.Adam(model_wav2v.parameters(), lr=args.wav2v_lr, betas=(0.9, 0.999))
	optim_sync = optim.Adam(model_sync.parameters(), lr=args.sync_lr, betas=(0.9, 0.999))
	criterion_class = nn.CrossEntropyLoss()
	sch_lmk2lip = optim.lr_scheduler.ExponentialLR(optim_lmk2lip, gamma=args.lmk2lip_gamma)
	sch_wav2v = optim.lr_scheduler.ExponentialLR(optim_wav2v, gamma=args.wav2v_gamma)
	sch_sync = optim.lr_scheduler.ExponentialLR(optim_sync, gamma=args.sync_gamma)
	tosave_list = ['model_lmk2lip', 'model_wav2v', 'model_sync',
	               'optim_lmk2lip', 'optim_wav2v', 'optim_sync',
	               'sch_lmk2lip', 'sch_wav2v', 'sch_sync']
	if args.wandb:
		for model_iter in model_list:
			wandb.watch(model_iter)

	# ============================度量载入===============================
	epoch_loss_class = Meter('Sync Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.4f')
	epoch_acc_sync = Meter('Sync ACC', 'avg', ':.2f', '%, ')
	epoch_timer = Meter('Time', 'time', ':4.0f')

	epoch_reset_list = [epoch_loss_final, epoch_loss_class,
	                    epoch_acc_sync,
	                    epoch_timer, ]
	if args.mode in ('train', 'continue'):
		print('Train Parameters')
		for key, value in args.__dict__.items():
			print(f'{key:18}:\t{value}')
		print('')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('%sStart loading dataset%s'%('='*20, '='*20))
	loader_timer.set_start_time(time.time())
	train_loader = LRWDataLoader(args.train_list, batch_size,
	                             num_workers=args.num_workers,
	                             n_mfcc=args.n_mfcc,
	                             resolution=args.resolution,
	                             seq_len=0,
	                             is_train=True, max_size=0)

	valid_loader = LRWDataLoader(args.train_list, batch_size,
	                             num_workers=args.num_workers,
	                             n_mfcc=args.n_mfcc,
	                             resolution=args.resolution,
	                             seq_len=0,
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
			with torch.no_grad():
				model_lmk2lip.eval()
				model_wav2v.eval()
				model_sync.eval()
				criterion_class.eval()
				evaluate(model_lmk2lip, model_wav2v, model_sync,
				         criterion_class,
				         valid_loader, args)
		else:
			del valid_loader
			test_loader = LRWImageTripletDataLoader(args.test_list, batch_size,
			                                        num_workers=args.num_workers,
			                                        seq_len=0, resolution=0,
			                                        is_train=False, max_size=0)
			with torch.no_grad():
				model_lmk2lip.eval()
				model_wav2v.eval()
				model_sync.eval()
				criterion_class.eval()
				evaluate(model_lmk2lip, model_wav2v, model_sync,
				         criterion_class,
				         test_loader, args)
		print(f'Finish Evaluation\n\n')
		return
	elif args.mode.lower() in ['continue']:
		print('Loading pretrained model', args.pretrain_model)
		model_ckpt = torch.load(args.pretrain_model)
		for item_str in tosave_list:
			item_model = locals()[item_str]
			item_model.load_state_dict(model_ckpt[item_str])
		start_epoch = model_ckpt['epoch']
		file_train_log = open(path_train_log, 'a')
	elif args.mode.lower() in ['train']:
		file_train_log = open(path_train_log, 'w')
		if args.pretrain_model is not None:
			model_ckpt = torch.load(args.pretrain_model)
			model_lmk2lip.load_state_dict(model_ckpt['model_lmk2lip'])
			model_wav2v.load_state_dict(model_ckpt['model_wav2v'])
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
			a_wav, a_lmk, a_wid = data
			a_wav = a_wav.to(run_device)
			a_lmk = a_lmk.to(run_device)
			a_wid = a_wid.to(run_device)
			a_lip = model_lmk2lip(a_lmk)
			a_voice = model_wav2v(a_wav)
			# todo: 这里是要用综合特征，还是按帧划分的特征？
			# todo: 这里是否要让 a_lip和 a_voice特征相近？

			new_idx = get_rand_idx(args.batch_size)
			a_voice = a_voice[new_idx, :]
			label_gt = get_gt_label(a_wid, new_idx).to(run_device)
			label_pred = model_sync(a_lip, a_voice)

			# ======================计算唇部特征单词分类损失===========================
			loss_class = criterion_class(label_pred, label_gt)
			correct_num_class = torch.sum(torch.argmax(label_pred, dim=1) == label_gt).item()

			# ==========================反向传播===============================
			optim_lmk2lip.zero_grad()
			optim_wav2v.zero_grad()
			optim_sync.zero_grad()
			loss_final = loss_class
			loss_final.backward()
			optim_lmk2lip.step()
			optim_wav2v.step()
			optim_sync.step()

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

		sch_lmk2lip.step()
		sch_wav2v.step()
		sch_sync.step()
		print('')
		print(f'Current Model M2V Learning Rate is {sch_lmk2lip.get_last_lr()}')
		print(f'Current Model V2T Learning Rate is {sch_wav2v.get_last_lr()}')
		print(f'Current Model Sync Learning Rate is {sch_sync.get_last_lr()}')
		print('Epoch:', epoch, epoch_loss_final, epoch_acc_sync,
		      file=file_train_log)
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
			with torch.no_grad():
				# torch.no_grad()不能停止drop_out和 batch_norm，所以还是需要eval
				model_lmk2lip.eval()
				model_wav2v.eval()
				model_sync.eval()
				criterion_class.eval()
				try:
					log_dict.update(evaluate(model_lmk2lip, model_wav2v, model_sync,
					                         criterion_class, valid_loader, args))
				except:
					print('Evaluating Error')
				model_lmk2lip.train()
				model_wav2v.train()
				model_sync.train()
				criterion_class.train()

		if args.wandb:
			wandb.log(log_dict)
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
	file_train_log.close()
	if args.wandb:
		wandb.finish()


if __name__ == '__main__':
	main()

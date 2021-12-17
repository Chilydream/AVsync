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

from model.Voice2TModel import Voice2T_fc_Model
from model.VGGModel import VGGVoice
from utils.tensor_utils import PadSquare

sys.path.append('/root/ChineseDataset/AVsync/third_party/yolo')
sys.path.append('/root/ChineseDataset/AVsync/third_party/HRNet')

from utils.data_utils.LRWAudio import LRWAudioDataLoader
from utils.data_utils.LRWAudioTriplet import LRWAudioTripletDataLoader
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter


def evaluate(model_wav2v, model_v2t, criterion_class, criterion_triplet, loader, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	val_loss_class = Meter('Class Loss', 'avg', ':.4f')
	val_loss_triplet = Meter('Triplet Loss', 'avg', ':.4f')
	val_loss_final = Meter('Final Loss', 'avg', ':.4f')
	val_acc_class = Meter('Class ACC', 'avg', ':.2f', '%,')
	val_timer = Meter('Time', 'time', ':3.0f')
	val_timer.set_start_time(time.time())

	print('\tEvaluating Result:')
	for data in loader:
		a_data, p_data, n_data, p_wid, n_wid = data
		a_data = a_data.to(run_device)
		p_data = p_data.to(run_device)
		n_data = n_data.to(run_device)
		apn_data = torch.cat((a_data, p_data, n_data), dim=0)
		p_wid = p_wid.to(run_device)
		n_wid = n_wid.to(run_device)
		apn_wid = torch.cat((p_wid, p_wid, n_wid), dim=0)

		apn_voice = model_wav2v(apn_data)
		apn_pred = model_v2t(apn_voice)
		a_voice, p_voice, n_voice = torch.chunk(apn_voice, 3, dim=0)

		# ======================计算 Triplet损失===========================
		loss_triplet = criterion_triplet(a_voice, p_voice, n_voice)

		# ======================计算音频单词分类损失===========================
		loss_class = criterion_class(apn_pred, apn_wid)
		correct_num_class = torch.sum(torch.argmax(apn_pred, dim=1) == apn_wid).item()

		# ===========================更新度量==============================
		loss_final = args.class_lambda*loss_class+args.triplet_lambda*loss_triplet
		val_loss_class.update(loss_class.item())
		val_loss_triplet.update(loss_triplet.item())
		val_loss_final.update(loss_final.item())
		val_acc_class.update(correct_num_class*100/args.batch_size/3)
		val_timer.update(time.time())
		print(f'\r\t{val_timer}{val_loss_final}{val_acc_class}',
		      f'{val_loss_class}{val_loss_triplet}',
		      end='     ')
	val_log = {'val_loss_class': val_loss_class.avg,
	           'val_loss_triplet': val_loss_triplet.avg,
	           'val_loss_final': val_loss_final.avg,
	           'val_acc_class': val_acc_class.avg}
	return val_log


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/speech2text.yaml').parse()
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
		wandb.init(project=args.project_name, config=args)

	# ============================模型载入===============================
	print('%sStart loading model%s'%('='*20, '='*20))
	model_wav2v = VGGVoice(n_out=args.voice_emb, n_mfcc=args.n_mfcc)
	model_v2t = Voice2T_fc_Model(args.voice_emb, n_class=500)
	model_list = [model_wav2v, model_v2t]
	for model_iter in model_list:
		model_iter.to(run_device)
		model_iter.train()

	optim_wav2v = optim.Adam(model_wav2v.parameters(), lr=args.wav2v_lr, betas=(0.9, 0.999))
	optim_v2t = optim.Adam(model_v2t.parameters(), lr=args.v2t_lr, betas=(0.9, 0.999))
	criterion_class = nn.CrossEntropyLoss()
	criterion_triplet = nn.TripletMarginLoss(margin=args.triplet_margin)
	sch_wav2v = optim.lr_scheduler.ExponentialLR(optim_wav2v, gamma=args.wav2v_gamma)
	sch_v2t = optim.lr_scheduler.ExponentialLR(optim_v2t, gamma=args.v2t_gamma)
	tosave_list = ['model_wav2v', 'model_v2t', 'optim_wav2v', 'optim_v2t', 'sch_wav2v', 'sch_v2t']
	if args.wandb:
		for model_iter in model_list:
			wandb.watch(model_iter)

	# ============================度量载入===============================
	epoch_loss_class = Meter('Class Loss', 'avg', ':.4f')
	epoch_loss_triplet = Meter('Triplet Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.4f')
	epoch_acc_class = Meter('Class ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':3.0f')

	epoch_reset_list = [epoch_loss_final, epoch_timer, epoch_loss_class, epoch_acc_class, epoch_loss_triplet]
	print('Train Parameters')
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}')
	print('')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('%sStart loading dataset%s'%('='*20, '='*20))
	loader_timer.set_start_time(time.time())
	train_loader = LRWAudioTripletDataLoader(args.train_list, batch_size, args.num_workers,
	                                         n_mfcc=0, is_train=True, max_size=0)
	valid_loader = LRWAudioTripletDataLoader(args.val_list, batch_size, num_workers=args.num_workers,
	                                         n_mfcc=0, is_train=False, max_size=0)
	loader_timer.update(time.time())
	print('Finish loading dataset', loader_timer)

	# ========================预加载模型================================
	if args.mode.lower() in ['test', 'valid', 'eval', 'val', 'evaluate']:
		model_ckpt = torch.load(args.pretrain_model)
		for item_str in tosave_list:
			item_model = locals()[item_str]
			item_model.load_state_dict(model_ckpt[item_str])
		if args.mode.lower() in ['valid', 'val', 'eval', 'evaluate']:
			evaluate(model_wav2v, model_v2t, criterion_class, criterion_triplet, valid_loader, args)
		else:
			test_loader = LRWAudioTripletDataLoader(args.test_list, batch_size, num_workers=args.num_workers,
			                                        n_mfcc=0, is_train=False, max_size=0)
			evaluate(model_wav2v, model_v2t, criterion_class, criterion_triplet, test_loader, args)
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
	else:
		raise Exception(f"未定义训练模式{args.mode}")

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
			a_data, p_data, n_data, p_wid, n_wid = data
			a_data = a_data.to(run_device)
			p_data = p_data.to(run_device)
			n_data = n_data.to(run_device)
			apn_data = torch.cat((a_data, p_data, n_data), dim=0)
			p_wid = p_wid.to(run_device)
			n_wid = n_wid.to(run_device)
			apn_wid = torch.cat((p_wid, p_wid, n_wid), dim=0)

			apn_voice = model_wav2v(apn_data)
			apn_pred = model_v2t(apn_voice)
			a_voice, p_voice, n_voice = torch.chunk(apn_voice, 3, dim=0)

			# ======================计算 Triplet损失===========================
			loss_triplet = criterion_triplet(a_voice, p_voice, n_voice)

			# ======================计算音频单词分类损失===========================
			loss_class = criterion_class(apn_pred, apn_wid)
			correct_num_class = torch.sum(torch.argmax(apn_pred, dim=1) == apn_wid).item()

			# ==========================反向传播===============================
			optim_wav2v.zero_grad()
			optim_v2t.zero_grad()
			loss_final = args.class_lambda*loss_class+args.triplet_lambda*loss_triplet
			loss_final.backward()
			optim_wav2v.step()
			optim_v2t.step()

			# ==========计量更新============================
			epoch_acc_class.update(correct_num_class*100/batch_size/3)
			epoch_loss_triplet.update(loss_triplet.item())
			epoch_loss_class.update(loss_class.item())
			epoch_loss_final.update(loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:04d}/{len(train_loader):04d}  {epoch_timer}{epoch_loss_final}{epoch_acc_class}',
			      f' EMA ACC: {epoch_acc_class.avg_ema:.2f}%, ',
			      f'{epoch_loss_class}{epoch_loss_triplet}',
			      sep='', end='     ')

		sch_wav2v.step()
		sch_v2t.step()
		print('')
		print(f'Current Model WAV2V Learning Rate is {sch_wav2v.get_last_lr()}')
		print(f'Current Model V2T Learning Rate is {sch_v2t.get_last_lr()}')
		print('Epoch:', epoch, epoch_loss_final, epoch_acc_class,
		      file=file_train_log)
		log_dict = {'epoch': epoch,
		            epoch_loss_final.name: epoch_loss_final.avg,
		            epoch_loss_class.name: epoch_loss_class.avg,
		            epoch_loss_triplet.name: epoch_loss_triplet.avg,
		            epoch_acc_class.name: epoch_acc_class.avg}
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
			torch.save(ckpt_dict, cache_dir+"/model%09d.model"%epoch)

		# ===========================验证=======================
		if (epoch+1)%args.valid_step == 0:
			with torch.no_grad():
				# torch.no_grad()不能停止drop_out和 batch_norm，所以还是需要eval
				model_wav2v.eval()
				model_v2t.eval()
				criterion_class.eval()
				criterion_triplet.eval()
				log_dict.update(
					evaluate(model_wav2v, model_v2t, criterion_class, criterion_triplet, valid_loader, args))
				model_wav2v.train()
				model_v2t.train()
				criterion_class.train()
				criterion_triplet.train()

		if args.wandb:
			wandb.log(log_dict)
	file_train_log.close()
	if args.wandb:
		wandb.finish()


if __name__ == '__main__':
	main()

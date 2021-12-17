import math
import os
import platform
import time
import numpy as np
import torch
import torchsnooper
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


def evaluate(model_wav2v, model_s2t, criterion, loader, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	val_loss_class = Meter('Audio Class Loss', 'avg', ':.4f')
	val_loss_final = Meter('Final Loss', 'avg', ':.4f')
	val_acc_class = Meter('Audio Class ACC', 'avg', ':.2f', '%,')
	val_timer = Meter('Time', 'time', ':3.0f')
	val_timer.set_start_time(time.time())

	print('\tEvaluating Result:')
	for data in loader:
		wid_label = data[1]
		mfcc_data = data[0].to(run_device)
		voice_data = model_wav2v(mfcc_data)
		label_pred = model_s2t(voice_data)
		label_gt = wid_label.to(run_device)

		# ======================计算音脸匹配损失===========================
		loss_class = criterion(label_pred, label_gt)
		acc_avmatch = torch.sum(torch.argmax(label_pred, dim=1, keepdim=False) == label_gt).item()
		loss_final = loss_class

		# ===========================更新度量==============================
		val_loss_class.update(loss_class.item())
		val_loss_final.update(loss_final.item())
		val_acc_class.update(acc_avmatch*100/label_gt.shape[0])
		val_timer.update(time.time())
		print(f'\r\t{val_timer}{val_loss_final}{val_acc_class}',
		      end='     ')
	val_log = {'val_loss_class': val_loss_class.avg,
	           'val_loss_final': val_loss_final.avg,
	           'val_acc_class': val_acc_class.avg}
	return val_log


# @torchsnooper.snoop()
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
		wandb.init(project=args.project_name, config=args,
		           name=args.exp_num, group=args.exp_num)

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
	criterion = nn.CrossEntropyLoss()
	sch_wav2v = optim.lr_scheduler.ExponentialLR(optim_wav2v, gamma=args.wav2v_gamma)
	sch_v2t = optim.lr_scheduler.ExponentialLR(optim_v2t, gamma=args.v2t_gamma)
	tosave_list = ['model_wav2v', 'model_v2t', 'optim_wav2v', 'optim_v2t', 'sch_wav2v', 'sch_v2t']
	if args.wandb:
		for model_iter in model_list:
			wandb.watch(model_iter)

	# ============================度量载入===============================
	epoch_loss_class = Meter('Audio Class Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Audio Final Loss', 'avg', ':.4f')
	epoch_acc_class = Meter('Audio Class ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':3.0f')

	epoch_reset_list = [epoch_loss_final, epoch_timer, epoch_loss_class, epoch_acc_class]
	print('Train Parameters')
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}')
	print('')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('%sStart loading dataset%s'%('='*20, '='*20))
	loader_timer.set_start_time(time.time())
	train_loader = LRWAudioDataLoader(args.train_list, batch_size, args.num_workers,
	                                  n_mfcc=0, is_train=True, max_size=0)
	valid_loader = LRWAudioDataLoader(args.val_list, batch_size, num_workers=args.num_workers,
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
			evaluate(model_wav2v, model_v2t, criterion, valid_loader, args)
		else:
			test_loader = LRWAudioDataLoader(args.test_list, batch_size, num_workers=args.num_workers,
			                                 n_mfcc=0, is_train=False, max_size=0)
			evaluate(model_wav2v, model_v2t, criterion, test_loader, args)
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
			# wav_data = (batch,seq,n_mfcc)
			wid_label = data[1]
			wav_data = data[0].to(run_device)
			voice_data = model_wav2v(wav_data)
			# voice_data = (batch, voice_emb)

			label_pred = model_v2t(voice_data)
			label_gt = wid_label.to(run_device)

			# ======================计算音频单词分类损失===========================
			loss_class = criterion(label_pred, label_gt)
			batch_acc_class = torch.sum(torch.argmax(label_pred, dim=1, keepdim=False) == label_gt).item()

			# ======================计算脸脸匹配损失===========================
			pass

			# ==========================反向传播===============================
			optim_wav2v.zero_grad()
			optim_v2t.zero_grad()
			batch_loss_final = loss_class
			batch_loss_final.backward()
			optim_wav2v.step()
			optim_v2t.step()

			# ==========计量更新============================
			epoch_acc_class.update(batch_acc_class*100/label_gt.shape[0])
			epoch_loss_class.update(loss_class.item())
			epoch_loss_final.update(batch_loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:04d}/{len(train_loader):04d}  ',
			      f'{epoch_timer}{epoch_loss_final}{epoch_acc_class}',
			      f' EMA ACC: {epoch_acc_class.avg_ema:.2f}% ', sep='', end='     ')

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
				criterion.eval()
				log_dict.update(evaluate(model_wav2v, model_v2t, criterion, valid_loader, args))
				model_wav2v.train()
				model_v2t.train()
				criterion.train()

		if args.wandb:
			wandb.log(log_dict)
	file_train_log.close()
	if args.wandb:
		wandb.finish()


if __name__ == '__main__':
	main()

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

from model.Lip2TModel import Lip2T_fc_Model
from model.Voice2TModel import Voice2T_fc_Model
from model.VGGModel import VGG6_speech, ResLip, VGG6_lip
from utils.data_utils.LRWImageTriplet import LRWImageTripletDataLoader
from utils.tensor_utils import PadSquare

sys.path.append('/root/ChineseDataset/AVsync/third_party/yolo')
sys.path.append('/root/ChineseDataset/AVsync/third_party/HRNet')

from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter


def evaluate(model_img2lip, model_lip2t, criterion_class, criterion_triplet, loader, args):
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
		apn_data = torch.cat((a_data, p_data, n_data), dim=0)
		apn_data.transpose_(2, 1)
		apn_data = apn_data.to(run_device)
		apn_wid = torch.cat((p_wid, p_wid, n_wid), dim=0)
		apn_wid = apn_wid.to(run_device)

		apn_lip = model_img2lip(apn_data)
		apn_word = model_lip2t(apn_lip)

		# ======================计算 Triplet损失===========================
		loss_triplet = criterion_triplet(apn_data[:args.batch_size],
		                                 apn_data[args.batch_size:2*args.batch_size],
		                                 apn_data[2*args.batch_size:])

		# ======================计算唇部特征单词分类损失===========================
		loss_class = criterion_class(apn_word, apn_wid)
		correct_num_class = torch.sum(torch.argmax(apn_word, dim=1, keepdim=False) == apn_wid).item()

		loss_final = args.class_lambda*loss_class+args.triplet_lambda*loss_triplet

		# ==========计量更新============================
		val_acc_class.update(correct_num_class*100/args.batch_size/3)
		val_loss_class.update(loss_class.item())
		val_loss_triplet.update(loss_triplet.item())
		val_loss_final.update(loss_final.item())
		val_timer.update(time.time())
		print(f'\r\tBatch:{val_timer.count:04d}/{len(loader):04d}  {val_timer}{val_loss_final}',
		      f'{val_acc_class} EMA ACC: {val_acc_class.avg_ema:.2f}%, ',
		      f'{val_loss_class}{val_loss_triplet}',
		      sep='', end='     ')

	val_log = {'val_loss_class': val_loss_class.avg,
	           'val_loss_triplet': val_loss_triplet.avg,
	           'val_loss_final': val_loss_final.avg,
	           'val_acc_class': val_acc_class.avg}
	return val_log


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/lip2text.yaml').parse()
	start_epoch = 0
	batch_size = args.batch_size
	torch.backends.cudnn.benchmark = args.gpu
	run_device = torch.device("cuda" if args.gpu else "cpu")

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
	# model_img2lip = ResLip(n_out=args.lip_emb)
	model_img2lip = VGG6_lip(n_out=args.lip_emb, stride=args.stride)
	model_lip2t = Lip2T_fc_Model(args.lip_emb, n_class=500)
	model_list = [model_img2lip, model_lip2t]
	for model_iter in model_list:
		model_iter.to(run_device)
		model_iter.train()

	optim_img2lip = optim.Adam(model_img2lip.parameters(), lr=args.img2lip_lr, betas=(0.9, 0.999))
	optim_lip2t = optim.Adam(model_lip2t.parameters(), lr=args.lip2t_lr, betas=(0.9, 0.999))
	criterion_class = nn.CrossEntropyLoss()
	criterion_triplet = nn.TripletMarginLoss(margin=args.triplet_margin)
	sch_img2lip = optim.lr_scheduler.ExponentialLR(optim_img2lip, gamma=args.img2lip_gamma)
	sch_lip2t = optim.lr_scheduler.ExponentialLR(optim_lip2t, gamma=args.lip2t_gamma)
	tosave_list = ['model_img2lip', 'model_lip2t',
	               'optim_img2lip', 'optim_lip2t',
	               'sch_img2lip', 'sch_lip2t']
	if args.wandb:
		for model_iter in model_list:
			wandb.watch(model_iter)

	# ============================度量载入===============================
	epoch_loss_class = Meter('Class Loss', 'avg', ':.4f')
	epoch_loss_triplet = Meter('Triplet Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.4f')
	epoch_acc_class = Meter('Class ACC', 'avg', ':.2f', '%,')
	epoch_timer = Meter('Time', 'time', ':4.0f')

	epoch_reset_list = [epoch_loss_final, epoch_timer,
	                    epoch_loss_triplet, epoch_loss_class, epoch_acc_class]
	print('Train Parameters')
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}')
	print('')

	# ============================数据载入===============================
	loader_timer = Meter('Time', 'time', ':3.0f', end='')
	print('%sStart loading dataset%s'%('='*20, '='*20))
	loader_timer.set_start_time(time.time())
	train_loader = LRWImageTripletDataLoader(args.train_list, batch_size,
	                                         num_workers=args.num_workers,
	                                         resolution=args.resolution,
	                                         is_train=True, max_size=0)

	valid_loader = LRWImageTripletDataLoader(args.val_list, batch_size,
	                                         num_workers=args.num_workers,
	                                         resolution=args.resolution,
	                                         is_train=False, max_size=0)
	loader_timer.update(time.time())
	print(f'Batch Num in Train Loader: {len(train_loader)}')
	print('Finish loading dataset', loader_timer)

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
				model_img2lip.eval()
				model_lip2t.eval()
				criterion_class.eval()
				criterion_triplet.eval()
				evaluate(model_img2lip, model_lip2t,
				         criterion_class, criterion_triplet,
				         valid_loader, args)
		else:
			del valid_loader
			test_loader = LRWImageTripletDataLoader(args.test_list, batch_size,
			                                        num_workers=args.num_workers,
			                                        seq_len=0, resolution=0,
			                                        is_train=False, max_size=0)
			with torch.no_grad():
				model_img2lip.eval()
				model_lip2t.eval()
				criterion_class.eval()
				criterion_triplet.eval()
				evaluate(model_img2lip, model_lip2t,
				         criterion_class, criterion_triplet,
				         test_loader, args)
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
			apn_data = torch.cat((a_data, p_data, n_data), dim=0)
			# apn_data = (3*batch, seq, 3, 256, 256)
			apn_data.transpose_(2, 1)
			# apn_data = (3*batch, 3, seq, 256, 256)
			apn_wid = torch.cat((p_wid, p_wid, n_wid), dim=0)

			apn_data = apn_data.to(run_device)
			apn_wid = apn_wid.to(run_device)

			apn_lip = model_img2lip(apn_data)
			apn_pred = model_lip2t(apn_lip)
			# ======================计算 Triplet损失===========================
			a_lip, p_lip, n_lip = torch.chunk(apn_lip, 3, dim=0)
			loss_triplet = criterion_triplet(a_lip, p_lip, n_lip)
			# loss_triplet = criterion_triplet(apn_data[:batch_size],
			#                                  apn_data[batch_size:2*batch_size],
			#                                  apn_data[2*batch_size:])

			# ======================计算唇部特征单词分类损失===========================
			loss_class = criterion_class(apn_pred, apn_wid)
			correct_num_class = torch.sum(torch.argmax(apn_pred, dim=1) == apn_wid).item()

			# ==========================反向传播===============================
			optim_img2lip.zero_grad()
			optim_lip2t.zero_grad()
			loss_final = args.class_lambda*loss_class+args.triplet_lambda*loss_triplet
			# torch.cuda.empty_cache()
			loss_final.backward()
			optim_img2lip.step()
			optim_lip2t.step()

			# ==========计量更新============================
			epoch_acc_class.update(correct_num_class*100/batch_size/3)
			epoch_loss_class.update(loss_class.item())
			epoch_loss_triplet.update(loss_triplet.item())
			epoch_loss_final.update(loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:04d}/{len(train_loader):04d}  {epoch_timer}{epoch_loss_final}',
			      f'{epoch_acc_class} EMA ACC: {epoch_acc_class.avg_ema:.2f}%, ',
			      f'{epoch_loss_class}{epoch_loss_triplet}',
			      sep='', end='     ')

		sch_img2lip.step()
		sch_lip2t.step()
		print('')
		print(f'Current Model M2V Learning Rate is {sch_img2lip.get_last_lr()}')
		print(f'Current Model V2T Learning Rate is {sch_lip2t.get_last_lr()}')
		log_dict = {'epoch': epoch,
		            epoch_loss_final.name: epoch_loss_final.avg,
		            epoch_loss_triplet.name: epoch_loss_triplet.avg,
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
			print(f'save model to {cache_dir+"/model%09d.model"%epoch}')
			torch.save(ckpt_dict, cache_dir+"/model%09d.model"%epoch)

		# ===========================验证=======================
		if args.valid_step>0 and (epoch+1)%args.valid_step == 0:
			with torch.no_grad():
				# torch.no_grad()不能停止drop_out和 batch_norm，所以还是需要eval
				model_img2lip.eval()
				model_lip2t.eval()
				criterion_class.eval()
				criterion_triplet.eval()
				try:
					log_dict.update(evaluate(model_img2lip, model_lip2t,
					                         criterion_class, criterion_triplet, valid_loader, args))
				except:
					print('Evaluating Error')
				model_img2lip.train()
				model_lip2t.train()
				criterion_class.train()

		if args.wandb:
			wandb.log(log_dict)
		print(log_dict, file=file_train_log)
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
	file_train_log.close()
	if args.wandb:
		wandb.finish()


if __name__ == '__main__':
	main()

import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
import sys

from model.SyncModel import SyncModel, SilenceModel
from utils.accuracy import get_gt_label, get_rand_idx
from utils.data_utils.LabLmk import LabLmkDataLoader
from utils.data_utils.LabLmkWav import LabLmkWavDataLoader
from utils.data_utils.LabRaw import LabDataLoader

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from model.Lmk2LipModel import Lmk2LipModel
from model.VGGModel import VGG6_speech
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks


def evaluate(model_lmk2lip, model_silence, criterion_class, loader, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	val_loss = Meter('Match Class Loss', 'avg', ':.4f')
	val_loss_final = Meter('Final Loss', 'avg', ':.4f')
	val_acc = Meter('Match Class ACC', 'avg', ':.2f', '%, ')
	val_timer = Meter('Time', 'time', ':3.0f')
	val_timer.set_start_time(time.time())

	print('\tEvaluating Result:')
	for data in loader:
		a_lmk, label_gt = data
		a_lmk = a_lmk.to(run_device)
		label_gt = label_gt.to(run_device)

		a_lip = model_lmk2lip(a_lmk)
		label_pred = model_silence(a_lip)

		# ======================计算唇部特征单词分类损失===========================
		loss_class = criterion_class(label_pred, label_gt)
		correct_num = torch.sum(torch.argmax(label_pred, dim=1) == label_gt).item()

		# ==========================反向传播===============================
		loss_final = loss_class

		# ==========计量更新============================
		val_acc.update(correct_num*100/len(label_gt))
		val_loss.update(loss_class.item())
		val_loss_final.update(loss_final.item())
		val_timer.update(time.time())
		print(f'\r\tBatch:{val_timer.count:04d}/{len(loader):04d}  {val_timer}{val_loss_final}',
		      f'{val_acc}',
		      sep='', end='     ')

	val_log = {'val_loss': val_loss.avg,
	           'val_loss_final': val_loss_final.avg,
	           'val_acc': val_acc.avg}
	return val_log


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/lab_silence.yaml').parse()
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
	print(f'{"="*20}Start loading model{"="*20}')

	model_lmk2lip = Lmk2LipModel(lmk_emb=args.lmk_emb, lip_emb=args.lip_emb, stride=1)
	model_silence = SilenceModel(lip_emb=args.lip_emb)
	model_list = [model_lmk2lip, model_silence]
	for model_iter in model_list:
		model_iter.to(run_device)
		model_iter.train()

	pad_resize = transforms.Compose([PadSquare(),
	                                 transforms.Resize(args.face_resolution)])

	optim_lmk2lip = optim.Adam(model_lmk2lip.parameters(), lr=args.lmk2lip_lr, betas=(0.9, 0.999))
	optim_silence = optim.Adam(model_silence.parameters(), lr=args.sync_lr, betas=(0.9, 0.999))
	criterion_class = nn.CrossEntropyLoss()
	sch_lmk2lip = optim.lr_scheduler.ExponentialLR(optim_lmk2lip, gamma=args.lmk2lip_gamma)
	sch_silence = optim.lr_scheduler.ExponentialLR(optim_silence, gamma=args.sync_gamma)
	tosave_list = ['model_lmk2lip', 'model_silence',
	               'optim_lmk2lip', 'optim_silence',
	               'sch_lmk2lip', 'sch_silence']
	if args.wandb:
		for model_iter in model_list:
			wandb.watch(model_iter)

	# ============================度量载入===============================
	epoch_loss_class = Meter('Match Class Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.4f')
	epoch_acc = Meter('Match Class ACC', 'avg', ':.2f', '%, ')
	epoch_timer = Meter('Time', 'time', ':4.0f')

	epoch_reset_list = [epoch_loss_final, epoch_loss_class,
	                    epoch_acc,
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
	train_loader = LabLmkDataLoader(args.train_list, batch_size,
	                                num_workers=args.num_workers,
	                                seq_len=args.seq_len,
	                                is_train=True, max_size=0)

	valid_loader = LabLmkDataLoader(args.val_list, batch_size,
	                                num_workers=args.num_workers,
	                                seq_len=args.seq_len,
	                                is_train=True, max_size=0)
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
				model_silence.eval()
				criterion_class.eval()
				evaluate(model_lmk2lip, model_silence,
				         criterion_class,
				         valid_loader, args)
		else:
			del valid_loader
			test_loader = LabLmkDataLoader(args.test_list, batch_size,
			                               num_workers=args.num_workers,
			                               seq_len=args.seq_len,
			                               is_train=True, max_size=0)
			with torch.no_grad():
				model_lmk2lip.eval()
				model_silence.eval()
				criterion_class.eval()
				evaluate(model_lmk2lip, model_silence,
				         criterion_class,
				         test_loader, args)
		print(f'\n\nFinish Evaluation\n\n')
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
		if args.pretrain_model is not None and os.path.exists(args.pretrain_model):
			model_ckpt = torch.load(args.pretrain_model)
			model_lmk2lip.load_state_dict(model_ckpt['model_lmk2lip'])
			if 'model_silence' in model_ckpt.keys():
				model_silence.load_state_dict(model_ckpt['model_silence'])
	else:
		raise Exception(f"未定义训练模式{args.mode}")

	print('Train Parameters', file=file_train_log)
	for key, value in args.__dict__.items():
		print(f'{key:18}:\t{value}', file=file_train_log)
	print('\n', file=file_train_log)

	# ============================开始训练===============================
	print('%sStart Training%s'%('='*20, '='*20))

	label_zero = torch.zeros(args.batch_size, dtype=torch.long, device=run_device)
	label_one = torch.ones(args.batch_size, dtype=torch.long, device=run_device)

	for epoch in range(start_epoch, args.epoch):
		print(f'\nEpoch: {epoch}')
		batch_cnt = 0
		epoch_timer.set_start_time(time.time())
		for data in train_loader:
			a_lmk, label_gt = data
			a_lmk = a_lmk.to(run_device)
			label_gt = label_gt.to(run_device)

			a_lip = model_lmk2lip(a_lmk)

			label_pred = model_silence(a_lip)

			# ======================计算唇部特征单词分类损失===========================
			loss_class = criterion_class(label_pred, label_gt)
			correct_num = torch.sum(torch.argmax(label_pred, dim=1) == label_gt).item()

			# ==========================反向传播===============================
			optim_lmk2lip.zero_grad()
			optim_silence.zero_grad()
			loss_final = loss_class
			loss_final.backward()
			optim_lmk2lip.step()
			optim_silence.step()

			# ==========计量更新============================
			epoch_acc.update(correct_num*100/len(label_gt))
			epoch_loss_class.update(loss_class.item())
			epoch_loss_final.update(loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:04d}/{len(train_loader):04d}  {epoch_timer}{epoch_loss_final}',
			      f'{epoch_acc}',
			      sep='', end='     ')
			torch.cuda.empty_cache()

		sch_lmk2lip.step()
		sch_silence.step()
		print('')
		print(f'Current Model M2V Learning Rate is {sch_lmk2lip.get_last_lr()}')
		print(f'Current Model Sync Learning Rate is {sch_silence.get_last_lr()}')
		print('Epoch:', epoch, epoch_loss_final, epoch_acc,
		      file=file_train_log)
		log_dict = {'epoch': epoch,
		            epoch_loss_final.name: epoch_loss_final.avg,
		            epoch_loss_class.name: epoch_loss_class.avg,
		            epoch_acc.name: epoch_acc.avg}
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
				model_silence.eval()
				criterion_class.eval()
				# try:
				log_dict.update(evaluate(model_lmk2lip, model_silence,
				                         criterion_class, valid_loader, args))
				# except:
				# 	print('Evaluating Error')
				model_lmk2lip.train()
				model_silence.train()
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

import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
import sys

from model.SyncModel import SyncModel
from utils.accuracy import get_gt_label, get_rand_idx
from utils.data_utils.LabLmk import LabLmkDataLoader
from utils.data_utils.LabRaw import LabDataLoader

sys.path.append('/home/tliu/fsx/project/AVsync/third_party/yolo')
sys.path.append('/home/tliu/fsx/project/AVsync/third_party/HRNet')

from model.Lmk2LipModel import Lmk2LipModel
from model.VGGModel import VGGVoice
from utils.tensor_utils import PadSquare
from utils.GetConsoleArgs import TrainOptions
from utils.Meter import Meter
from third_party.yolo.yolo_models.yolo import Model as yolo_model
from third_party.yolo.yolo_utils.util_yolo import face_detect
from third_party.HRNet.utils_inference import get_model_by_name, get_batch_lmks


def evaluate(model_lmk2lip, model_wav2v, model_sync, criterion_class, loader, args):
	run_device = torch.device("cuda:0" if args.gpu else "cpu")

	val_loss_gt = Meter('Match Class Loss', 'avg', ':.4f')
	val_loss_fk = Meter('Mismatch Class Loss', 'avg', ':.4f')
	val_loss_final = Meter('Final Loss', 'avg', ':.4f')
	val_acc_gt = Meter('Match Class ACC', 'avg', ':.2f', '%, ')
	val_acc_fk = Meter('Mismatch Class ACC', 'avg', ':.2f', '%, ')
	val_acc_all = Meter('All Class ACC', 'avg', ':.2f', '%, ')
	val_timer = Meter('Time', 'time', ':3.0f')
	val_timer.set_start_time(time.time())
	label_one = torch.ones(args.batch_size, dtype=torch.long, device=run_device)
	label_zero = torch.zeros(args.batch_size, dtype=torch.long, device=run_device)

	print('\tEvaluating Result:')
	for data in loader:
		a_lmk, a_wav_gt, a_wav_fk = data
		a_lmk = a_lmk.to(run_device)
		a_wav_gt = a_wav_gt.to(run_device)
		a_wav_fk = a_wav_fk.to(run_device)

		a_lip = model_lmk2lip(a_lmk)
		a_voice_gt = model_wav2v(a_wav_gt)
		a_voice_fk = model_wav2v(a_wav_fk)

		label_pred_gt = model_sync(a_lip, a_voice_gt)
		label_pred_fk = model_sync(a_lip, a_voice_fk)
		print(f'\npred gt \n{label_pred_gt}\n')
		print(f'\npred fk \n{label_pred_fk}\n')

		# ======================计算唇部特征单词分类损失===========================
		loss_class_gt = criterion_class(label_pred_gt, label_one)
		loss_class_fk = criterion_class(label_pred_fk, label_zero)
		correct_num_gt = torch.sum(torch.argmax(label_pred_gt, dim=1) == label_one).item()
		correct_num_fk = torch.sum(torch.argmax(label_pred_fk, dim=1) == label_zero).item()

		loss_final = loss_class_gt+loss_class_fk

		# ==========计量更新============================
		val_acc_gt.update(correct_num_gt*100/len(label_one))
		val_acc_gt.update(correct_num_fk*100/len(label_zero))
		val_acc_all.update((val_acc_gt.avg+val_acc_fk.avg)*0.5)
		val_loss_gt.update(loss_class_gt.item())
		val_loss_fk.update(loss_class_fk.item())
		val_loss_final.update(loss_final.item())
		val_timer.update(time.time())
		print(f'\r\tBatch:{val_timer.count:04d}/{len(loader):04d}  {val_timer}{val_loss_final}',
		      f'{val_acc_all}{val_acc_gt}{val_acc_fk}',
		      sep='', end='     ')

	val_log = {'val_loss_gt': val_loss_gt.avg,
	           'val_loss_final': val_loss_final.avg,
	           'val_acc_gt': val_acc_gt.avg}
	return val_log


def main():
	# ===========================参数设定===============================
	args = TrainOptions('config/lab_sync.yaml').parse()
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
	model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
	model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt'))
	model_yolo.to(run_device)
	model_hrnet = get_model_by_name('300W', root_models_path='pretrain_model')
	model_hrnet = model_hrnet.eval()
	model_hrnet.to(run_device)

	model_lmk2lip = Lmk2LipModel(lmk_emb=args.lmk_emb, lip_emb=args.lip_emb, stride=1)
	model_wav2v = VGGVoice(n_out=args.voice_emb)
	model_sync = SyncModel(lip_emb=args.lip_emb, voice_emb=args.voice_emb)
	model_list = [model_lmk2lip, model_wav2v, model_sync]
	for model_iter in model_list:
		model_iter.to(run_device)
		model_iter.train()

	pad_resize = transforms.Compose([PadSquare(),
	                                 transforms.Resize(args.face_resolution)])

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
	epoch_loss_class_gt = Meter('Match Class Loss', 'avg', ':.4f')
	epoch_loss_class_fk = Meter('Mismatch Class Loss', 'avg', ':.4f')
	epoch_loss_final = Meter('Final Loss', 'avg', ':.4f')
	epoch_acc_gt = Meter('Match Class ACC', 'avg', ':.2f', '%, ')
	epoch_acc_fk = Meter('Mismatch Class ACC', 'avg', ':.2f', '%, ')
	epoch_acc_all = Meter('All Class ACC', 'avg', ':.2f', '%, ')
	epoch_timer = Meter('Time', 'time', ':4.0f')

	epoch_reset_list = [epoch_loss_final, epoch_loss_class_gt,
	                    epoch_acc_gt,
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
				model_wav2v.eval()
				model_sync.eval()
				criterion_class.eval()
				evaluate(model_lmk2lip, model_wav2v, model_sync,
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
				model_wav2v.eval()
				model_sync.eval()
				criterion_class.eval()
				evaluate(model_lmk2lip, model_wav2v, model_sync,
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
		if args.pretrain_model is not None:
			model_ckpt = torch.load(args.pretrain_model)
			model_lmk2lip.load_state_dict(model_ckpt['model_lmk2lip'])
			model_wav2v.load_state_dict(model_ckpt['model_wav2v'])
			if 'model_sync' in model_ckpt.keys():
				model_sync.load_state_dict(model_ckpt['model_sync'])
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
			a_lmk, a_wav_gt, a_wav_fk = data
			a_lmk = a_lmk.to(run_device)
			a_wav_gt = a_wav_gt.to(run_device)
			a_wav_fk = a_wav_fk.to(run_device)
			# a_lmk = (b, seq_len, 40)
			# a_wav = (b, seq_len*16000)

			a_lip = model_lmk2lip(a_lmk)
			a_voice_gt = model_wav2v(a_wav_gt)
			a_voice_fk = model_wav2v(a_wav_fk)

			label_pred_gt = model_sync(a_lip, a_voice_gt)
			label_pred_fk = model_sync(a_lip, a_voice_fk)
			# print(f'\npred gt \n{label_pred_gt}\n')
			# print(f'\npred fk \n{label_pred_fk}\n')

			# ======================计算唇部特征单词分类损失===========================
			loss_class_gt = criterion_class(label_pred_gt, label_one)
			loss_class_fk = criterion_class(label_pred_fk, label_zero)
			correct_num_gt = torch.sum(torch.argmax(label_pred_gt, dim=1) == label_one).item()
			correct_num_fk = torch.sum(torch.argmax(label_pred_fk, dim=1) == label_zero).item()

			# ==========================反向传播===============================
			optim_lmk2lip.zero_grad()
			optim_wav2v.zero_grad()
			optim_sync.zero_grad()
			loss_final = loss_class_gt+loss_class_fk
			loss_final.backward()
			optim_lmk2lip.step()
			optim_wav2v.step()
			optim_sync.step()

			# ==========计量更新============================
			epoch_acc_gt.update(correct_num_gt*100/len(label_pred_gt))
			epoch_acc_fk.update(correct_num_fk*100/len(label_pred_fk))
			epoch_acc_all.update((epoch_acc_gt.avg+epoch_acc_fk.avg)*0.5)
			epoch_loss_class_gt.update(loss_class_gt.item())
			epoch_loss_class_fk.update(loss_class_fk.item())
			epoch_loss_final.update(loss_final.item())
			epoch_timer.update(time.time())
			batch_cnt += 1
			print(f'\rBatch:{batch_cnt:04d}/{len(train_loader):04d}  {epoch_timer}{epoch_loss_final}',
			      f'{epoch_acc_all}{epoch_acc_gt}{epoch_acc_fk}',
			      sep='', end='     ')
			torch.cuda.empty_cache()

		sch_lmk2lip.step()
		sch_wav2v.step()
		sch_sync.step()
		print('')
		print(f'Current Model M2V Learning Rate is {sch_lmk2lip.get_last_lr()}')
		print(f'Current Model V2T Learning Rate is {sch_wav2v.get_last_lr()}')
		print(f'Current Model Sync Learning Rate is {sch_sync.get_last_lr()}')
		print('Epoch:', epoch, epoch_loss_final, epoch_acc_gt,
		      file=file_train_log)
		log_dict = {'epoch': epoch,
		            epoch_loss_final.name: epoch_loss_final.avg,
		            epoch_loss_class_gt.name: epoch_loss_class_gt.avg,
		            epoch_acc_gt.name: epoch_acc_gt.avg}
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
				# try:
				log_dict.update(evaluate(model_lmk2lip, model_wav2v, model_sync,
				                         criterion_class, valid_loader, args))
				# except:
				# 	print('Evaluating Error')
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

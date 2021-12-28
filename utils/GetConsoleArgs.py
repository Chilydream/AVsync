import argparse
import os
import platform
from glob import glob
import torch
import yaml


def str2bool(string):
	return string.lower()=='true'


class TrainOptions:
	def __init__(self, train_config):
		parser = argparse.ArgumentParser(description='Audio Video Synchronization')
		if isinstance(train_config, dict):
			for key, value in train_config.items():
				arg_type = type(value) if value is not None else str
				default_value = value
				if arg_type==bool:
					arg_type=str2bool
					default_value = 'true' if value else 'false'
				parser.add_argument(f'--{key}', default=default_value,
				                    type=arg_type)
		elif isinstance(train_config, str):
			a = yaml.load(open(train_config, 'r'), Loader=yaml.FullLoader)
			for key, value in a.items():
				arg_type = type(value) if type(value) is not None else str
				default_value = value
				if arg_type==bool:
					arg_type=str2bool
					default_value = 'true' if value else 'false'
				parser.add_argument(f'--{key}', default=default_value,
				                    type=arg_type)
		self.args = parser.parse_args()

	def parse(self):
		self.args.mode = self.args.mode.lower()
		self.args.wandb = self.args.wandb and (self.args.mode in {'train', 'continue'})
		self.args.gpu = self.args.gpu and torch.cuda.is_available()
		if platform.system() == 'Windows':
			self.args.num_workers = 0
			self.args.train_list = "metadata/desktop/train.txt"
			self.args.val_list = "metadata/desktop/val.txt"
			self.args.test_list = "metadata/desktop/test.txt"

		if self.args.pretrain_model is not None and \
				not os.path.exists(self.args.pretrain_model):
			print(f'警告：找不到预训练文件{self.args.pretrain_model}')
			self.args.pretrain_model = None

		if self.args.exp_dir is None:
			self.args.exp_dir = 'exp'
		if self.args.exp_num is None or int(self.args.exp_num) == 0:
			if not os.path.exists(self.args.exp_dir):
				self.args.exp_num = '001'
			else:
				exp_list = os.listdir(self.args.exp_dir)
				exp_list.sort()
				newest_exp_num = exp_list[-1]
				self.args.exp_num = f'{int(newest_exp_num)+1:03d}'

		cur_exp = os.path.join(self.args.exp_dir, self.args.exp_num)
		# if os.path.exists(cur_exp) and not self.args.ddp_flag:
		if os.path.exists(cur_exp):
			model_dir = os.path.join(self.args.exp_dir, self.args.exp_num, 'cache', 'model0*.model')
			model_list = glob(model_dir)
			model_list.sort()
			if len(model_list)>0 and self.args.mode == 'train':
				raise Exception(f"实验编号`{self.args.exp_num}`已被占用，"
				                f"请删除旧的实验记录或者使用新的实验编号")
			if self.args.pretrain_model is None and\
					self.args.mode in ('continue', 'eval', 'evaluate', 'test', 'val'):
				if len(model_list)>0:
					self.args.pretrain_model = model_list[-1]
					print(f'警告：没有指定预训练模型或指定的预训练模型不存在，'
					      f'自动使用{self.args.exp_num}号实验保存的最新预训练模型`{model_list[-1]}`')

		if self.args.mode in ('continue', 'eval', 'evaluate', 'test', 'val') and\
				self.args.pretrain_model is None:
			raise Exception(f'训练模式是`{self.args.mode}`，但是没有指定可用的预训练模型')
		return self.args


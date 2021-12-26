import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block2(nn.Module):
	def __init__(self, input_feature, output_feature, kernel_size, stride=None, padding=None):
		super(Block2, self).__init__()
		stride = stride if stride is not None else kernel_size
		padding = padding if padding is not None else list(map(int, kernel_size/2))

		self.res = None
		if stride!=1 and input_feature==output_feature:
			self.res = nn.MaxPool2d(kernel_size=1, stride=stride)
		elif stride!=1:
			self.res = nn.Conv2d(in_channels=input_feature, out_channels=output_feature,
			                     kernel_size=1, stride=stride)

		self.model0 = nn.Sequential(
			nn.Conv2d(in_channels=input_feature, out_channels=output_feature,
			          kernel_size=kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(output_feature),
			nn.ReLU(True),
			nn.Conv2d(in_channels=output_feature, out_channels=output_feature,
			          kernel_size=kernel_size, stride=1, padding=padding),
		)
		self.model1 = nn.Sequential(
			nn.BatchNorm2d(output_feature),
			nn.ReLU(),
		)

	def forward(self, x):
		res = x if self.res is None else self.res(x)
		x = self.model0(x)
		x = self.model1(x+res)
		return x


class Block3(nn.Module):
	def __init__(self, input_feature, output_feature, kernel_size, stride=None, padding=None):
		super(Block3, self).__init__()
		stride = stride if stride is not None else kernel_size
		padding = padding if padding is not None else list(map(int, kernel_size/2))

		self.res = None
		if stride!=1 and input_feature==output_feature:
			self.res = nn.MaxPool3d(kernel_size=1, stride=stride)
		elif stride!=1 or input_feature!=output_feature:
			self.res = nn.Conv3d(in_channels=input_feature, out_channels=output_feature,
			                     kernel_size=1, stride=stride)

		self.model0 = nn.Sequential(
			nn.Conv3d(in_channels=input_feature, out_channels=output_feature,
			          kernel_size=kernel_size, stride=stride, padding=self.padding),
			nn.BatchNorm3d(output_feature),
			nn.ReLU(True),
			nn.Conv3d(in_channels=output_feature, out_channels=output_feature,
			          kernel_size=kernel_size, stride=1, padding=self.padding),
		)
		self.model1 = nn.Sequential(
			nn.BatchNorm3d(output_feature),
			nn.ReLU(),
		)

	def forward(self, x):
		res = x if self.res is None else self.res(x)
		x = self.model0(x)
		x = self.model1(x+res)
		return x


class MultiSensory(nn.Module):
	def __init__(self, sound_rate=16000, image_fps=25):
		super(MultiSensory, self).__init__()
		# 音频输入先进行简单的归一化
		# tf.sign(sfs)*(tf.log(1 + scale*tf.abs(sfs)) / tf.log(1 + scale))

		# todo: 音频网络的 padding需要计算一遍
		self.snd_net0 = nn.Sequential(
			# 要求输入是双声道
			# (b, c=2, snd_len=44144, 1)
			nn.Conv2d(2, 64, kernel_size=(65, 1), stride=(4, 1), padding=(32, 0)),
			# (b, 64, 11036, 1)
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
			# (b, 64, 2759, 1)
		)
		self.snd_res1 = nn.Sequential(
			# (b, 64, 2759, 1)
			nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(4, 1)),
			# (b, 128, 690, 1)
			nn.BatchNorm2d(128),
		)
		self.snd_net1 = nn.Sequential(
			# (b, 64, 2759, 1)
			nn.Conv2d(64, 128, kernel_size=(15, 1), stride=(4, 1), padding=(7, 0)),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.Conv2d(128, 128, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
			# (b, 128, 690, 1)
			# 然后加上 snd_res1
		)
		self.snd_bn1 = nn.Sequential(
			nn.BatchNorm3d(128),
			nn.ReLU(),
		)
		self.snd_res2 = nn.Sequential(
			# (b, 128, 690, 1)
			nn.MaxPool2d(kernel_size=(1, 1), stride=(4, 1)),
			# (b, 128, 173, 1)
		)
		self.snd_net2 = nn.Sequential(
			# (b, 128, 690, 1)
			nn.Conv2d(128, 128, kernel_size=(15, 1), stride=(4, 1), padding=(7, 0)),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.Conv2d(128, 128, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
			# (b, 128, 173, 1)
			# 然后加上 snd_res2
		)
		self.snd_bn2 = nn.Sequential(
			nn.BatchNorm3d(128),
			nn.ReLU(),
		)
		self.snd_res3 = nn.Sequential(
			# (b, 128, 173, 1)
			nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(4, 1)),
			# (b, 256, 44, 1)
			nn.BatchNorm2d(256),
		)
		self.snd_net3 = nn.Sequential(
			# (b, 128, 173, 1)
			nn.Conv2d(128, 256, kernel_size=(15, 1), stride=(4, 1), padding=(7, 0)),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size=(15, 1), stride=(1, 1), padding=(7, 0)),
			# (b, 256, 44, 1)
			# 然后加上 snd_res3
		)
		self.snd_bn3 = nn.Sequential(
			nn.BatchNorm3d(256),
			nn.ReLU(),
		)

		# 输入的图片也要进行一次归一化
		# -1. + (2./255) * im
		self.img_net0 = nn.Sequential(
			# x = (b, 3, 63, 224, 224)
			nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3)),
			# x = (b, 64, 32, 112, 112)
			nn.BatchNorm3d(64),
			nn.ReLU(True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
			# x = (b, 64, 32, 56, 56)
		)
		# todo: img_res1 = img_net0的输出
		self.img_net1 = nn.Sequential(
			# x = (b, 64, 32, 56, 56)
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),  # 2-1-1
			nn.BatchNorm3d(64),
			nn.ReLU(True),
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),  # 2-1-2
			# x = (b, 64, 32, 56, 56)
			# 然后加 img_res1
		)
		self.img_bn1 = nn.Sequential(
			nn.BatchNorm3d(64),
			nn.ReLU(),
		)
		self.img_res2 = nn.Sequential(
			# x = (b, 64, 32, 56, 56)
			nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 2)),
			# x = (b, 64, 16, 28, 28)
		)
		self.img_net2 = nn.Sequential(
			# x = (b, 64, 32, 56, 56)
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
			# x = (b, 64, 16, 28, 28)
			nn.BatchNorm3d(64),
			nn.ReLU(True),
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
			# x = (b, 64, 16, 28, 28)
			# 然后加 img_res2
		)
		self.img_bn2 = nn.Sequential(
			nn.BatchNorm3d(64),
			nn.ReLU(),
		)

		# img_num = fps/4
		# snd_num = rate/1024
		self.frac_pool = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(image_fps*256/sound_rate))
		# 要将 (b, 256, 44, 1) 转换成 (1, 256, 16, 1)
		# 将输入的音频和视频帧对应上
		# ques：kernel_size要设置为多少？
		# ques：torch的这个网络层可能很不好用
		self.snd_net4 = nn.Sequential(
			# (b, 256, 16, 1)
			nn.Conv2d(256, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),  # sf/conv5_1
			# (b, 128, 16, 1)
			nn.BatchNorm2d(128),
			nn.ReLU(),
		)
		# 取单声道，然后将维度修改为 (b, 128, 16, 1, 1)
		# 再使用 torch.repeat 将维度修改为 (b, 128, 16, 28, 28)也就是和img的大小相同
		# 拼接 音频特征和图像特征，得到 (b, 256, 16, 28, 28)
		# todo: 取 (b, :64, ...)和 (b, -64:, ...) 拼接得到残差向量 merge_res

		self.merge_conv = nn.Sequential(
			nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
			nn.BatchNorm3d(512),
			nn.ReLU(True),
			nn.Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
			# (b, 128, 16, 28, 28)
			# 然后和 merge_res相加
		)
		self.merge_bn = nn.Sequential(
			nn.BatchNorm3d(128),
			nn.ReLU(),
		)

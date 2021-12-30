import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

from utils.tensor_utils import FracPool


# @torchsnooper.snoop()
class Block2(nn.Module):
	def __init__(self, input_feature, output_feature, kernel_size, stride=None, padding=None):
		super(Block2, self).__init__()
		stride = stride if stride is not None else kernel_size
		padding = padding if padding is not None else list(map(lambda x: max(0, int(x/2)), kernel_size))

		self.res = None
		if stride != 1 and input_feature == output_feature:
			self.res = nn.MaxPool2d(kernel_size=1, stride=stride)
		elif stride != 1:
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
		padding = padding if padding is not None else list(map(lambda x: max(0, int(x/2)), kernel_size))

		self.res = None
		if stride != 1 and input_feature == output_feature:
			self.res = nn.MaxPool3d(kernel_size=1, stride=stride)
		elif stride != 1 or input_feature != output_feature:
			self.res = nn.Conv3d(in_channels=input_feature, out_channels=output_feature,
			                     kernel_size=1, stride=stride)

		self.model0 = nn.Sequential(
			nn.Conv3d(in_channels=input_feature, out_channels=output_feature,
			          kernel_size=kernel_size, stride=stride, padding=padding),
			nn.BatchNorm3d(output_feature),
			nn.ReLU(True),
			nn.Conv3d(in_channels=output_feature, out_channels=output_feature,
			          kernel_size=kernel_size, stride=1, padding=padding),
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


# @torchsnooper.snoop()
class MultiSensory(nn.Module):
	def __init__(self, sound_rate=21000, image_fps=30, audio_channel=1):
		super(MultiSensory, self).__init__()
		# 音频输入先进行简单的归一化
		# tf.sign(sfs)*(tf.log(1 + scale*tf.abs(sfs)) / tf.log(1 + scale))

		self.snd_pre = nn.Sequential(
			# 要求输入是双声道
			# (b, audio_channel, snd_len=44144, 1)
			nn.Conv2d(audio_channel, 64, kernel_size=(65, 1), stride=(4, 1), padding=(32, 0)),
			# (b, 64, 11036, 1)
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
			# (b, 64, 2759, 1)
		)
		self.snd_block0 = Block2(64, 128, kernel_size=(15, 1), stride=(4, 1))
		self.snd_block1 = Block2(128, 128, kernel_size=(15, 1), stride=(4, 1))
		self.snd_block2 = Block2(128, 256, kernel_size=(15, 1), stride=(4, 1))

		# 输入的图片也要进行一次归一化
		# -1. + (2./255) * im
		self.img_pre = nn.Sequential(
			# x = (b, 3, 63, 224, 224)
			nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3)),
			# x = (b, 64, 32, 112, 112)
			nn.BatchNorm3d(64),
			nn.ReLU(True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
			# x = (b, 64, 32, 56, 56)
		)
		self.img_block0 = Block3(64, 64, kernel_size=(3, 3, 3), stride=1)
		self.img_block1 = Block3(64, 64, kernel_size=(3, 3, 3), stride=2)

		# img_num = fps/4
		# snd_num = rate/1024
		# self.frac_pool = nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(image_fps*256/sound_rate))
		# self.frac_pool = nn.FractionalMaxPool2d(kernel_size=3, output_size=(8, 1))
		self.frac_pool = FracPool(sound_rate/image_fps/256)

		# 要将 (b, 256, 44, 1) 转换成 (b, 256, 16, 1)
		# 将输入的音频和视频帧对应上
		self.snd_net4 = nn.Sequential(
			# (b, 256, 16, 1)
			nn.Conv2d(256, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),  # sf/conv5_1
			# (b, 128, 16, 1)
			nn.BatchNorm2d(128),
		)
		# 取单声道，然后将维度修改为 (b, 128, 16, 1, 1)
		# 再使用 torch.repeat 将维度修改为 (b, 128, 16, 28, 28)也就是和img的大小相同
		# 拼接 音频特征和图像特征，得到 (b, 192, 16, 28, 28)
		# 取 (b, :64, ...)和 (b, -64:, ...) 拼接得到残差向量 merge_res

		self.merge_conv = nn.Sequential(
			nn.Conv3d(192, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
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

		self.merge_block0 = Block3(128, 128, kernel_size=(3, 3, 3), stride=1)  # 3-1
		self.merge_block1 = Block3(128, 128, kernel_size=(3, 3, 3), stride=1)  # 3-2
		self.merge_block2 = Block3(128, 256, kernel_size=(3, 3, 3), stride=2)  # 4-1
		self.merge_block3 = Block3(256, 256, kernel_size=(3, 3, 3), stride=1)  # 4-2
		# todo: 设置 time_stride
		time_stride = 1
		self.merge_block4 = Block3(256, 512, kernel_size=(3, 3, 3), stride=(time_stride, 2, 2))  # 5-1
		self.merge_block5 = Block3(512, 512, kernel_size=(3, 3, 3), stride=1)  # 5-2
		# (b, 512, 8, 7, 7)
		self.merge_mean = nn.AdaptiveAvgPool3d((1, 1, 1))
		# (b, 512, 1, 1, 1)
		self.joint_logits = nn.Sequential(
			nn.Conv3d(512, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
			# 然后取 x[:, :, 0, 0, 0]
		)
		self.joint_class = nn.Sequential(
			nn.Linear(1, 2),
			nn.Softmax(1),
		)
		self.cam = nn.Sequential(
			# 这里输入的是求均值前的 (b, 512, 8, 7, 7)
			nn.Conv3d(512, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
		)

	def snd_forward(self, snds):
		# (b, c=2, snd_len=44144, 1)
		snd_scale = 255
		snds = torch.sign(snds)*(torch.log(1+snd_scale*torch.abs(snds))/np.log(1+snd_scale))
		if snds.dim() == 2:
			snds.unsqueeze_(1)
			snds.unsqueeze_(-1)
		elif snds.dim() == 3:
			snds.unsqueeze_(-1)

		x = self.snd_pre(snds)
		x = self.snd_block0(x)
		x = self.snd_block1(x)
		x = self.snd_block2(x)
		return x

	def img_forward(self, imgs):
		# (b, c=3, img_len=63, 224, 224)
		# 如果图片的值是 0~255就需要对图片进行归一化
		imgs = (2./255)*imgs-1.0

		x = self.img_pre(imgs)
		x = self.img_block0(x)
		x = self.img_block1(x)
		return x

	def img_to_word(self, img_feature):
		# img_feature = (b, 64, 16, 28, 28)
		pass

	# @torchsnooper.snoop()
	def merge_forward(self, snd_feature, img_feature):
		# snd_feature = (b, 256, 44, 1)
		# img_feature = (b, 64, 16, 28, 28)
		snd_feature = self.frac_pool(snd_feature)
		# snd_feature = (b, 256, 16, 1)
		snd_feature = self.snd_net4(snd_feature)
		# snd_feature = (b, 128, 16, 1)
		snd_feature.unsqueeze_(4)
		snd_feature: torch.Tensor
		snd_feature = snd_feature.repeat((1, 1, 1, img_feature.shape[-2], img_feature.shape[-1]))
		# snd_feature = (b, 128, 16, 28, 28)
		snd_feature = F.relu(snd_feature)

		merge_feature = torch.cat((snd_feature, img_feature), dim=1)
		# merge_feature = (b, 192, 16, 28, 28)
		merge_res = torch.cat((merge_feature[:, :64, ...], merge_feature[:, -64:, ...]), dim=1)
		# merge_res = (b, 128, 16, 28, 28)

		merge_feature = self.merge_conv(merge_feature)
		merge_feature = self.merge_bn(merge_feature+merge_res)
		merge_feature = self.merge_block0(merge_feature)
		merge_feature = self.merge_block1(merge_feature)
		merge_feature = self.merge_block2(merge_feature)
		merge_feature = self.merge_block3(merge_feature)
		merge_feature = self.merge_block4(merge_feature)
		merge_feature = self.merge_block5(merge_feature)
		# (b, 512, 8, 7, 7)
		merge_mean = self.merge_mean(merge_feature)
		# (b, 512, 1, 1, 1)
		joint_logit = self.joint_logits(merge_mean)
		# (b, 1, 1, 1, 1)
		joint_logit = joint_logit[:, :, 0, 0, 0]
		# joint_logit = (b, 1)
		joint_class = self.joint_class(joint_logit)
		return joint_class

	def forward(self):
		raise NotImplementedError

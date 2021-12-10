import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
import torchsnooper


class SyncModel(nn.Module):
	# 一分钟大约能说 160~180 个汉字
	# 每个汉字平均发音时长约为 350ms
	# 假设视频帧数是 25fps，则每个汉字大约对应 9帧
	def __init__(self, lip_emb, voice_emb, seq_len=29):
		super(SyncModel, self).__init__()
		self.lip_emb = lip_emb
		self.voice_emb = voice_emb
		self.seq_len = seq_len

		# x = (b, seq, lip_emb+voice_emb)
		self.model = nn.Sequential(
			nn.Linear(lip_emb+voice_emb, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Linear(128, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.Linear(64, 32),
			nn.BatchNorm1d(32),
			nn.ReLU(True),
			nn.Linear(32, 2),
			nn.Softmax(-1),
		)

	def forward(self, lip, voice):
		# x.shpae = (batch, lip_emb+voice_emb)
		x = torch.cat((lip, voice), dim=1)
		x = self.model(x)
		return x

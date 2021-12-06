import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
import torchsnooper


# @torchsnooper.snoop()
class SpeakModel(nn.Module):
	# 一分钟大约能说 160~180 个汉字
	# 每个汉字平均发音时长约为 350ms
	# 假设视频帧数是 25fps，则每个汉字大约对应 9帧
	def __init__(self, face_emb, hid_emb=512, seq_len=16, batch_first=True):
		assert batch_first
		super(SpeakModel, self).__init__()
		input_size = face_emb
		self.batch_first = batch_first
		self.seq_len = seq_len
		self.hid_emb = hid_emb
		self.relu = nn.ReLU(inplace=True)
		self.tanh = nn.Tanh()
		self.bn0 = nn.BatchNorm1d(seq_len)

		self.lstm = nn.LSTM(input_size=input_size,
		                    hidden_size=hid_emb,
		                    batch_first=batch_first,
		                    bidirectional=True)
		self.bn1 = nn.BatchNorm1d(2*hid_emb)
		self.hn_fc1 = nn.Linear(2*hid_emb, 128)
		self.hn_fc2 = nn.Linear(128, 32)
		self.hn_fc3 = nn.Linear(32, 1)

	def forward(self, x):
		# todo: 是用含所有帧信息的 output，还是用只有最后一帧信息的 hn或cn？
		# x.shpae = (seq_len, batch, face_emb)
		x = self.bn0(x)
		output, (hn, cn) = self.lstm(x)

		hn = hn.transpose(1, 0).contiguous().view(-1, 2*self.hid_emb)
		hn = self.bn1(hn)
		hn = self.relu(self.hn_fc1(hn))
		hn = self.relu(self.hn_fc2(hn))
		hn = self.tanh(self.hn_fc3(hn))
		hn = hn.squeeze()
		return hn
import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
import torchsnooper


# @torchsnooper.snoop()
class SyncModel1(nn.Module):
	# 一分钟大约能说 160~180 个汉字
	# 每个汉字平均发音时长约为 350ms
	# 假设视频帧数是 25fps，则每个汉字大约对应 9帧
	def __init__(self, face_emb, voice_emb, hid_emb=512, seq_len=29, batch_first=True):
		super(SyncModel1, self).__init__()
		input_size = face_emb+voice_emb
		self.batch_first = batch_first
		self.seq_len = seq_len
		self.hid_emb = hid_emb
		self.relu = nn.ReLU(inplace=True)
		self.tanh = nn.Tanh()

		self.lstm = nn.LSTM(input_size=input_size,
		                    hidden_size=hid_emb,
		                    batch_first=batch_first,
		                    bidirectional=True)
		# self.output_fc1 = nn.Linear(in_features=hid_emb*2,
		#                             out_features=1)
		# self.output_fc2 = nn.Linear(in_features=seq_len,
		#                             out_features=1,
		#                             bias=False)
		self.hn_fc1 = nn.Linear(2*hid_emb, 128)
		self.hn_fc2 = nn.Linear(128, 32)
		self.hn_fc3 = nn.Linear(32, 1)

	def forward(self, face, voice):
		# todo: 是用含所有帧信息的 output，还是用只有最后一帧信息的 hn或cn？
		# x.shpae = (seq_len, batch, face_emb+voice_emb)
		x = torch.cat((face, voice), dim=2)
		output, (hn, cn) = self.lstm(x)
		#
		# hn = hn.transpose(1, 0).contiguous().view(-1, 2*self.hid_emb)
		# hn = self.relu(self.hn_fc1(hn))
		# hn = self.relu(self.hn_fc2(hn))
		# hn = self.tanh(self.hn_fc3(hn))
		# hn = hn.squeeze()
		# return hn

		if self.batch_first:
			batch_size, seq_len, feature_size = output.shape
		else:
			seq_len, batch_size, feature_size = output.shape

		output = output.view((-1, feature_size))
		output = self.output_fc1(output)
		if self.batch_first:
			output = output.view((batch_size, seq_len))
		else:
			output = output.view((seq_len, batch_size)).permute(1, 0)
		# output = (batch_size, seq_len)
		output = self.output_fc2(output)
		output = self.tanh(output)
		output = output.squeeze()
		return output


class SyncModel2(nn.Module):
	def __init__(self, lip_emb, voice_emb, seq_len=29):
		super(SyncModel2, self).__init__()
		self.lip_emb = lip_emb
		self.voice_emb = voice_emb
		self.seq_len = seq_len

		# x = (b, seq, lip_emb+voice_emb)
		self.model = nn.Sequential(nn.Linear(lip_emb+voice_emb, 512),
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
		                           nn.Tanh(),
		                           )

	def forward(self, lip, voice):
		# x.shpae = (batch, lip_emb+voice_emb)
		x = torch.cat((lip, voice), dim=1)
		x = self.model(x)
		return x

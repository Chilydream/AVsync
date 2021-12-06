import torch
import torch.nn as nn
import torchaudio
import torchsnooper
import torchvision


class Lmk2LipModel(nn.Module):
	def __init__(self, lmk_emb=40, lip_emb=256, stride=1):
		super(Lmk2LipModel, self).__init__()
		self.lmk_emb = lmk_emb
		self.lip_emb = lip_emb
		self.stride = stride
		# input = (b, seq, lmk_emb)
		self.lmk2lip = nn.Sequential(
			nn.Linear(lmk_emb, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Linear(128, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(True),
			nn.Linear(512, lip_emb)
		)
		# lip = (b, seq, lip_emb)
		# lip_T = (b, lip_emb, seq)
		self.frame5to1 = nn.Conv1d(lip_emb, lip_emb, kernel_size=(5,), stride=(stride,))
		# mid = (b, lip_emb, seq-4)
		self.final_mix = nn.Sequential(
			nn.Conv1d(lip_emb, lip_emb, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(lip_emb),
			nn.ReLU(True),
			nn.Conv1d(lip_emb, lip_emb, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(lip_emb),
			nn.ReLU(True),
			nn.Conv1d(lip_emb, lip_emb, kernel_size=(3,), stride=(2,)),
			nn.AdaptiveAvgPool1d(1),
		)

	def forward(self, x):
		batch_size, seq_len = x.shape[0], x.shape[1]
		lip = self.lmk2lip(x.reshape(batch_size*seq_len, self.lmk_emb))
		lip = lip.view(batch_size, seq_len, -1)
		lip.transpose_(2, 1)
		mid = self.frame5to1(lip)
		out = self.final_mix(mid).squeeze()
		return out

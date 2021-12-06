import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torchvision
import torchsnooper


# @torchsnooper.snoop()
class Lip2T_fc_Model(nn.Module):
	def __init__(self, face_emb, n_class):
		super(Lip2T_fc_Model, self).__init__()
		self.face_emb = face_emb
		self.n_class = n_class

		self.fc = nn.Sequential(nn.Linear(face_emb, 256),
		                        nn.BatchNorm1d(256),
		                        nn.ReLU(True),
		                        nn.Linear(256, 512),
		                        nn.BatchNorm1d(512),
		                        nn.ReLU(True),
		                        nn.Linear(512, 512),
		                        nn.BatchNorm1d(512),
		                        nn.ReLU(True),
		                        nn.Linear(512, n_class),
		                        nn.Softmax(-1)
		                        )

	def forward(self, x):
		out = self.fc(x)
		return out

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSensory(nn.Module):
	def __init__(self):
		super(MultiSensory, self).__init__()
		self.img_net0 = nn.Sequential(
			# x = (b, 3, 63, 224, 224)
			nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3)),
			# x = (b, 64, 32, 112, 112)
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
			# x = (b, 64, 32, 56, 56)
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
			# x = (b, 64, 32, 56, 56)
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
			# x = (b, 64, 32, 56, 56)
		)
		self.res_pool0 = nn.Sequential(
			# x = (b, 64, 32, 56, 56)
			nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 2)),
			# x = (b, 64, 16, 28, 28)
		)
		self.img_net1 = nn.Sequential(
			# x = (b, 64, 32, 56, 56)
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
			# x = (b, 64, 16, 28, 28)
			nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
			# x = (b, 64, 16, 28, 28)
		)


import torch
import torch.nn as nn
import os
import numpy as np
import torchvision


class FaceModel(nn.Module):
	def __init__(self, n_out, pretrained='pretrain_model/res34_partial.pt'):
		super(FaceModel, self).__init__()
		self.resnet34 = torchvision.models.resnet34(num_classes=n_out)
		# todo: 也可以考虑额外再加一层fc层
		if pretrained is not None:
			if os.path.exists(pretrained):
				self.resnet34.load_state_dict(torch.load(pretrained), strict=False)
			else:
				print(f'警告：你所指定的FaceModel的预训练模型`{pretrained}`不存在')

	def forward(self, x: torch.Tensor):
		assert x.dim() in (4, 5), ValueError(f"Dimension of FaceModel input should be 4 or 5, but get{x.dim()}")
		dim_flag = x.dim()==5
		if dim_flag:
			d0, d1 = x.shape[:2]
			x.view(-1, x.shape[2], x.shape[3], x.shape[4])
		x = self.resnet34(x)
		if dim_flag:
			x.view(d0, d1, x.shape[2], x.shape[3], x.shape[4])
		return x

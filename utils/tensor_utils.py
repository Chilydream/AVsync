import torch
import torch.nn as nn
import torch.nn.functional as F


class PadSquare(nn.Module):
	def __init__(self, pad_code=0):
		super(PadSquare, self).__init__()
		if not isinstance(pad_code, int):
			raise TypeError(f"In PadSquare Transformers, type(pad code) should be int, but get {type(pad_code)}")
		self.pad_code = pad_code

	def forward(self, img):
		s = max(img.shape[-2:])
		new_shape = list(img.shape[:-2])
		new_shape.extend([s, s])
		output = torch.full(new_shape, self.pad_code, dtype=torch.float32)
		new_y, new_x = (s-img.shape[-2])//2, (s-img.shape[-1])//2
		output[:, new_y:new_y+img.shape[-2], new_x:new_x+img.shape[-1]] = img
		return output


class MyContrastiveLoss(nn.Module):
	def __init__(self, margin=1.0):
		self.margin = margin
		super(MyContrastiveLoss, self).__init__()

	def forward(self, feature0, feature1, label):
		euclidean_distance = F.pairwise_distance(feature0, feature1, keepdim=True)
		loss = torch.mean(label*torch.pow(euclidean_distance, 2)+
		                  (1-label)*torch.pow(torch.clamp(self.margin-euclidean_distance, min=0.0), 2))
		return loss

import torch
import torch.nn as nn
import torchaudio
import torchsnooper
import torchvision

from model.FaceModel import FaceModel


class VGGVoice(nn.Module):
	def __init__(self, n_out=512, stride=1, n_mfcc=40):
		super(VGGVoice, self).__init__()
		self.vgg = nn.Sequential(
			# (b, 1, 40, 127)
			nn.Conv2d(1, 96, kernel_size=(5, 7), stride=(1, 1), padding=(2, 2)),
			# (b, 96, 40, 125)
			nn.BatchNorm2d(96),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
			# (b, 96, 40, 62)

			nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 1), padding=(1, 1)),
			# (b, 256, 19, 60)
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
			# (b, 256, 9, 58)

			nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
			# (b, 256, 9, 58)
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
			# (b, 256, 9, 58)
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
			# (b, 256, 9, 58)
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
			# (b, 256, 4, 28) for hop_length=160
			# (b, 256, 4, 25) for hop_length=175

			nn.Conv2d(256, 512, kernel_size=(4, 1), padding=(0, 0), stride=(1, stride)),
			# (b, 512, 1, 25) for hop_length=175
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

		self.fc = nn.Sequential(
			nn.Conv1d(512, 512, kernel_size=(1,)),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Conv1d(512, n_out, kernel_size=(1,)),
		)

		self.mix = nn.Sequential(
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(n_out),
			nn.ReLU(),
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(n_out),
			nn.ReLU(),
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(n_out),
			nn.AdaptiveAvgPool1d(1),
		)

		self.instancenorm = nn.InstanceNorm1d(40)
		self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
		                                                    hop_length=175, f_min=0.0, f_max=8000,
		                                                    pad=0, n_mels=n_mfcc)

	def forward(self, x):
		x = self.torchfb(x)+1e-6
		# x = (batch, n_mels=40, 129)
		x = self.instancenorm(x.log())
		x = x[:, :, 1:-1].detach()
		# x = (b, n_mels=40, 127)

		mid = self.vgg(x.unsqueeze(1))
		mid = mid.view((mid.size()[0], mid.size()[1], -1))
		# mid = (b, 512, 25) for hop_length=175

		emb_seq = self.fc(mid)
		# emb_seq = (b, nOut, 25) for hop_length=175

		emb_mix = self.mix(emb_seq).squeeze()
		return emb_mix


class ResLip(nn.Module):
	def __init__(self, n_out=256, stride=1):
		super(ResLip, self).__init__()
		# self.resnet = torchvision.models.resnet34(num_classes=n_out)
		# resnet_pretrain = 'pretrain_model/res34_partial.pt'
		# self.resnet.load_state_dict(torch.load(resnet_pretrain), strict=False)
		self.resnet = nn.Sequential(torchvision.models.resnet18(pretrained=True),
		                            nn.Linear(1000, n_out),
		                            )
		self.frame5to1 = nn.Sequential(
			nn.Conv1d(n_out, n_out, kernel_size=(5,), stride=(stride,), padding=0),
		)

		self.emb2word = nn.Sequential(
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.ReLU(),
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.ReLU(),
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.AdaptiveAvgPool1d(1),
		)

	def forward(self, x):

		# x = (b, 29, 3, 256, 256)
		d0, d1 = x.shape[:2]
		lip_emb = self.resnet(x.view(-1, *x.shape[2:]))
		lip_emb = lip_emb.view(d0, d1, -1)
		lip_emb.transpose_(2, 1)
		# lip_emb = (b, feature, 29)
		lip_5in1 = self.frame5to1(lip_emb)
		# lip_5in1 = (b, feature, 29-4)
		word_emb = self.emb2word(lip_5in1).squeeze()
		return word_emb


class VGGLip(nn.Module):
	def __init__(self, n_out=256, stride=1):
		super(VGGLip, self).__init__()
		self.vgg = nn.Sequential(
			# (4, 3, 29, 256, 256)
			nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(stride, 2, 2), padding=0),
			# (4, 96, 25, 125, 125)
			nn.BatchNorm3d(96),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
			# (4, 96, 25, 62, 62)

			nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
			# (4, 256, 25, 30, 30)
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
			# (4, 256, 25, 15, 15)

			nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
			# (4, 256, 25, 15, 15)
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),

			nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
			# (4, 256, 25, 15, 15)
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),

			nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
			# (4, 256, 25, 15, 15)
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
			# (4, 256, 25, 7, 7)

			nn.Conv3d(256, 512, kernel_size=(1, 7, 7), padding=0),
			# (4, 512, 25, 1, 1)
			nn.BatchNorm3d(512),
			nn.ReLU(inplace=True),
			# nn.AdaptiveAvgPool2d(1),
		)

		self.fc = nn.Sequential(
			nn.Conv1d(512, 512, kernel_size=(1,)),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Conv1d(512, n_out, kernel_size=(1,))
		)

		self.mix = nn.Sequential(
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(n_out),
			nn.ReLU(),
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(n_out),
			nn.ReLU(),
			nn.Conv1d(n_out, n_out, kernel_size=(3,), stride=(2,)),
			nn.BatchNorm1d(n_out),
			nn.AdaptiveAvgPool1d(1),
		)

	def forward(self, x):
		# x = (5, 3, 29, 256, 256)
		mid = self.vgg(x)
		mid = mid.view((mid.size()[0], mid.size()[1], -1))  # N x (ch x 24)
		# mid = (4, 512, 25)
		emb_seq = self.fc(mid)
		# emb_seq = (5, nOut, 25)
		emb_mix = self.mix(emb_seq).squeeze()
		return emb_mix

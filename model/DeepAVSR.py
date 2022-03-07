"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResNetLayer(nn.Module):
	"""
	A ResNet layer used to build the ResNet network.
	Architecture:
	--> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
	 |                        |   |                                    |
	 -----> downsample ------>    ------------------------------------->
	"""

	def __init__(self, inplanes, outplanes, stride):
		super(ResNetLayer, self).__init__()
		self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
		self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
		self.stride = stride
		self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False)
		self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

		self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
		self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
		self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
		return

	def forward(self, inputBatch):
		batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
		batch = self.conv2a(batch)
		if self.stride == 1:
			residualBatch = inputBatch
		else:
			residualBatch = self.downsample(inputBatch)
		batch = batch+residualBatch
		intermediateBatch = batch
		batch = F.relu(self.outbna(batch))

		batch = F.relu(self.bn1b(self.conv1b(batch)))
		batch = self.conv2b(batch)
		residualBatch = intermediateBatch
		batch = batch+residualBatch
		outputBatch = F.relu(self.outbnb(batch))
		return outputBatch


class ResNet(nn.Module):
	"""
	An 18-layer ResNet architecture.
	"""

	def __init__(self):
		super(ResNet, self).__init__()
		self.layer1 = ResNetLayer(64, 64, stride=1)
		self.layer2 = ResNetLayer(64, 128, stride=2)
		self.layer3 = ResNetLayer(128, 256, stride=2)
		self.layer4 = ResNetLayer(256, 512, stride=2)
		self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))
		return

	def forward(self, inputBatch):
		batch = self.layer1(inputBatch)
		batch = self.layer2(batch)
		batch = self.layer3(batch)
		batch = self.layer4(batch)
		outputBatch = self.avgpool(batch)
		return outputBatch


class VisualFrontend(nn.Module):
	"""
	A visual feature extraction module. Generates a 512-dim feature vector per video frame.
	Architecture: A 3D convolution block followed by an 18-layer ResNet.
	"""

	def __init__(self):
		super(VisualFrontend, self).__init__()
		self.frontend3D = nn.Sequential(
			nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
			nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
		)
		self.resnet = ResNet()
		return

	def forward(self, inputBatch):
		inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
		batchsize = inputBatch.shape[0]
		batch = self.frontend3D(inputBatch)

		batch = batch.transpose(1, 2)
		batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
		outputBatch = self.resnet(batch)
		outputBatch = outputBatch.reshape(batchsize, -1, 512)
		outputBatch = outputBatch.transpose(1, 2)
		outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
		return outputBatch


class PositionalEncoding(nn.Module):
	"""
	A layer to add positional encodings to the inputs of a Transformer model.
	Formula:
	PE(pos,2i) = sin(pos/10000^(2i/d_model))
	PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
	"""

	def __init__(self, dModel, maxLen):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(maxLen, dModel)
		position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
		denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
		pe[:, 0::2] = torch.sin(position/denominator)
		pe[:, 1::2] = torch.cos(position/denominator)
		pe = pe.unsqueeze(dim=0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self, inputBatch):
		outputBatch = inputBatch+self.pe[:inputBatch.shape[0], :, :]
		return outputBatch


class AVNet(nn.Module):
	"""
	An audio-visual speech transcription model based on the Transformer architecture.
	Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
				  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
				  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
	Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
	Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
				 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
	Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
	Output: Log probabilities over the character set at each time step.
	"""

	def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
		super(AVNet, self).__init__()
		self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
		self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
		encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize,
		                                          dropout=dropout)
		self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.jointConv = nn.Conv1d(2*dModel, dModel, kernel_size=1, stride=1, padding=0)
		self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
		self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
		return

	def forward(self, inputBatch):
		audioInputBatch, videoInputBatch = inputBatch

		if audioInputBatch is not None:
			audioInputBatch = audioInputBatch.transpose(0, 1).transpose(1, 2)
			audioBatch = self.audioConv(audioInputBatch)
			audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
			audioBatch = self.positionalEncoding(audioBatch)
			audioBatch = self.audioEncoder(audioBatch)
		else:
			audioBatch = None

		if videoInputBatch is not None:
			videoBatch = self.positionalEncoding(videoInputBatch)
			videoBatch = self.videoEncoder(videoBatch)
		else:
			videoBatch = None

		if (audioBatch is not None) and (videoBatch is not None):
			jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
			jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
			jointBatch = self.jointConv(jointBatch)
			jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
		elif (audioBatch is None) and (videoBatch is not None):
			jointBatch = videoBatch
		elif (audioBatch is not None) and (videoBatch is None):
			jointBatch = audioBatch
		else:
			print("Both audio and visual inputs missing.")
			exit()

		jointBatch = self.jointDecoder(jointBatch)
		jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
		jointBatch = self.outputConv(jointBatch)
		jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
		outputBatch = F.log_softmax(jointBatch, dim=2)
		return outputBatch

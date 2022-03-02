import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet18

from model.MultiSensory import MultiSensory
from utils.GetConsoleArgs import TrainOptions
from utils.GetDataFromFile import get_frame_and_wav_cv2
from utils.data_utils.LabRaw import LabDataLoader


class GradCAM:
	def __init__(self, model: nn.Module, target_layer: str, size=(224, 224), num_cls=1000, mean=None, std=None) -> None:
		self.model = model
		self.model.eval()

		# register hook
		# 可以自己指定层名，没必要一定通过target_layer传递参数
		# self.model.layer4
		# self.model.layer4[1].register_forward_hook(self.__forward_hook)
		# self.model.layer4[1].register_backward_hook(self.__backward_hook)
		getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
		getattr(self.model, target_layer).register_backward_hook(self.__backward_hook)

		self.size = size
		self.origin_size = None
		self.num_cls = num_cls

		self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
		if mean and std:
			self.mean, self.std = mean, std

		self.grads = []
		self.fmaps = []

	def forward(self, vid_tchw, wav_tensor, label=None, output_name='out/camcam.mp4'):
		# forward
		self.origin_time = vid_tchw.shape[1]
		vid_cthw = vid_tchw.transpose(2, 1)
		a_lip = self.model.img_forward(vid_cthw)
		a_wav = self.model.snd_forward(wav_tensor)
		a_pred = self.model.merge_forward(snd_feature=a_wav, img_feature=a_lip)

		# backward
		self.origin_size = vid_tchw.shape[-2:]
		vid_arr = vid_tchw.squeeze(0).numpy()
		self.model.zero_grad()
		loss = self.__compute_loss(a_pred, label)
		print(a_pred[0, 1]>a_pred[0, 0])
		loss.backward()

		# generate CAM
		grads_val = self.grads[0].cpu().data.numpy().squeeze()
		fmap = self.fmaps[0].cpu().data.numpy().squeeze()
		cam_list = self.__compute_cam(fmap, grads_val)
		org_list = []
		for i in range(len(cam_list)):
			cam_list[i] = cv2.resize(cam_list[i], self.origin_size)
			org_list.append(vid_arr[i].astype(np.float32)/255)
		self.__show_cam_on_image(org_list, cam_list, output_name)

		self.fmaps.clear()
		self.grads.clear()

	def __backward_hook(self, module, grad_in, grad_out):
		self.grads.append(grad_out[0].detach())

	def __forward_hook(self, module, input, output):
		self.fmaps.append(output)

	def __compute_loss(self, logit, label=None):
		if not label:
			label = np.argmax(logit.cpu().data.numpy())
		else:
			label = np.array(label)

		label = label[np.newaxis, np.newaxis]
		label = torch.from_numpy(label)
		one_hot = torch.zeros(1, self.num_cls)
		one_hot[0, label] = 1
		one_hot.requires_grad = True
		loss = torch.sum(one_hot*logit)
		return loss

	def __compute_cam(self, feature_map: np.ndarray, grads):
		"""
		feature_map: np.array [C, T, H, W]
		grads: np.array, [C, T, H, W]
		return: np.array, [T, H, W]
		"""
		grads = grads.transpose((1, 0, 2, 3))
		feature_map = feature_map.transpose((1, 0, 2, 3))
		time_length = feature_map.shape[0]
		cam_list = []
		for i in range(time_length):
			cam = np.zeros(feature_map.shape[2:], dtype=np.float32)
			alpha = np.mean(grads[i], axis=(1, 2))  # GAP
			for k, ak in enumerate(alpha):
				cam += ak*feature_map[i, k]  # linear combination

			cam = np.maximum(cam, 0)  # relu
			cam = cv2.resize(cam, self.size)
			cam = (cam-np.min(cam))/np.max(cam)
			cam_list.append(cam)
		j = 0
		frag = self.origin_time*1.0/(time_length-1)
		final_cam_list = []
		for i in range(time_length-1):
			while j<frag*(i+1):
				left_dist = (j-frag*i)/frag
				cam = left_dist*cam_list[i] + (1-left_dist)*cam_list[i+1]
				final_cam_list.append(cam)
				j += 1

		return final_cam_list

	def __show_cam_on_image(self, img_list: list, mask_list: list, output_name='out/camcam.mp4'):
		time_length = len(mask_list)
		video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, self.origin_size)
		for i in range(time_length):
			heatmap = cv2.applyColorMap(np.uint8(255*mask_list[i]), cv2.COLORMAP_JET)
			heatmap = np.float32(heatmap)/255
			img = np.float32(img_list[i])
			img = img.transpose(1, 2, 0)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			cam = heatmap+img
			cam = cam/np.max(cam)
			cam = np.uint8(255*cam)
			video.write(cam)
		video.release()


def main():
	args = TrainOptions('config/sync_multisensory.yaml').parse()
	model_ms = MultiSensory(sound_rate=16000, image_fps=25)
	grad_cam = GradCAM(model_ms, 'merge_block5', (224, 224), 2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

	model_ckpt = torch.load(args.pretrain_model, map_location='cpu')
	model_ms.load_state_dict(model_ckpt['model_ms'])
	train_loader = LabDataLoader(args.train_list, 1,
	                             num_workers=args.num_workers,
	                             tgt_frame_num=args.tgt_frame_num,
	                             tgt_fps=args.tgt_fps,
	                             resolution=args.img_size,
	                             wav_hz=16000,
	                             avspeech_flag=args.tmp_flag,
	                             is_train=False, )
	# valid_loader = LabDataLoader(args.val_list, 1,
	#                              num_workers=args.num_workers,
	#                              tgt_frame_num=args.tgt_frame_num,
	#                              tgt_fps=args.tgt_fps,
	#                              resolution=args.img_size,
	#                              wav_hz=16000,
	#                              avspeech_flag=args.tmp_flag,
	#                              is_train=False, )
	test_loader = LabDataLoader(args.test_list, 1,
	                             num_workers=args.num_workers,
	                             tgt_frame_num=args.tgt_frame_num,
	                             tgt_fps=args.tgt_fps,
	                             resolution=args.img_size,
	                             wav_hz=16000,
	                             avspeech_flag=args.tmp_flag,
	                             is_train=False, )

	for idx, data in enumerate(test_loader):
		a_img, a_wav_match = data
		grad_cam.forward(a_img, a_wav_match, label=1, output_name=f'out/LRW_match_{idx}.mp4')
		if idx>=9:
			break
	wav_mis = None
	for idx, data in enumerate(train_loader):
		a_img, a_wav_match = data
		if wav_mis is None:
			wav_mis = a_wav_match
		grad_cam.forward(a_img, a_wav_match, label=0, output_name=f'out/lab_match_{idx}.mp4')
		if idx>=9:
			break



# 调用函数
if __name__ == '__main__':
	main()

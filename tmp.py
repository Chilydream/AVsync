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

	def forward(self, img_arr: np.ndarray, label=None, show=True, write=False):
		img_input = self.__img_preprocess(img_arr.copy())

		# forward
		output = self.model(img_input)
		idx = np.argmax(output.cpu().data.numpy())

		# backward
		self.model.zero_grad()
		loss = self.__compute_loss(output, label)

		loss.backward()

		# generate CAM
		grads_val = self.grads[0].cpu().data.numpy().squeeze()
		fmap = self.fmaps[0].cpu().data.numpy().squeeze()
		cam = self.__compute_cam(fmap, grads_val)

		# show
		cam_show = cv2.resize(cam, self.origin_size)
		img_show = img_arr.astype(np.float32)/255
		self.__show_cam_on_image(img_show, cam_show, if_show=show, if_write=write)

		self.fmaps.clear()
		self.grads.clear()

	def my_forward(self, vid_arr, wav_arr, label=None, output_name='out/camcam.mp4'):
		# forward
		a_lip = self.model.img_forward(vid_arr)
		a_wav = self.model.snd_forward(wav_arr)
		a_pred = self.model.merge_forward(snd_feature=a_lip, img_feature=a_wav)

		# backward
		self.model.zero_grad()
		loss = self.__compute_loss(a_pred, label)
		loss.backward()

		# generate CAM
		grads_val = self.grads[0].cpu().data.numpy().squeeze()
		fmap = self.fmaps[0].cpu().data.numpy().squeeze()
		cam_list = self.__my_compute_cam(fmap, grads_val)
		org_list = []
		for i in range(len(cam_list)):
			cam_list[i] = cv2.resize(cam_list[i], self.origin_size)
			org_list.append(vid_arr[i].astype(np.float32)/255)
		self.__my_show_cam_on_image(org_list, cam_list, output_name)

		self.fmaps.clear()
		self.grads.clear()

	def __img_transform(self, img_arr: np.ndarray, transform: torchvision.transforms) -> torch.Tensor:
		img = img_arr.copy()  # [H, W, C]
		img = Image.fromarray(np.uint8(img))
		img = transform(img).unsqueeze(0)  # [N,C,H,W]
		return img

	def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
		self.origin_size = (img_in.shape[1], img_in.shape[0])  # [H, W, C]
		img = img_in.copy()
		img = cv2.resize(img, self.size)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(self.mean, self.std)
		])
		img_tensor = self.__img_transform(img, transform)
		return img_tensor

	def __backward_hook(self, module, grad_in, grad_out):
		self.grads.append(grad_out[0].detach())

	def __forward_hook(self, module, input, output):
		self.fmaps.append(output)

	def __compute_loss(self, logit, index=None):
		if not index:
			index = np.argmax(logit.cpu().data.numpy())
		else:
			index = np.array(index)

		index = index[np.newaxis, np.newaxis]
		index = torch.from_numpy(index)
		one_hot = torch.zeros(1, self.num_cls).scatter_(1, index, 1)
		one_hot.requires_grad = True
		loss = torch.sum(one_hot*logit)
		return loss

	def __compute_cam(self, feature_map, grads):
		"""
		feature_map: np.array [C, H, W]
		grads: np.array, [C, H, W]
		return: np.array, [H, W]
		"""
		cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
		alpha = np.mean(grads, axis=(1, 2))  # GAP
		for k, ak in enumerate(alpha):
			cam += ak*feature_map[k]  # linear combination

		cam = np.maximum(cam, 0)  # relu
		cam = cv2.resize(cam, self.size)
		cam = (cam-np.min(cam))/np.max(cam)
		return cam

	def __my_compute_cam(self, feature_map, grads):
		"""
		feature_map: np.array [T, C, H, W]
		grads: np.array, [T, C, H, W]
		return: np.array, [T, H, W]
		"""
		time_length = feature_map.shape[0]
		cam_list = []
		for i in range(time_length):
			cam = np.zeros(feature_map.shape[2:], dtype=np.float32)
			alpha = np.mean(grads, axis=(2, 3))  # GAP
			for k, ak in enumerate(alpha):
				cam += ak*feature_map[i, k]  # linear combination

			cam = np.maximum(cam, 0)  # relu
			cam = cv2.resize(cam, self.size)
			cam = (cam-np.min(cam))/np.max(cam)
			cam_list.append(cam)
		return cam_list

	def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, if_show=True, if_write=False):
		heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap)/255
		cam = heatmap+np.float32(img)
		cam = cam/np.max(cam)
		cam = np.uint8(255*cam)
		if if_write:
			cv2.imwrite("camcam.jpg", cam)
		if if_show:
			# 要显示RGB的图片，如果是BGR的 热力图是反过来的
			plt.imshow(cam[:, :, ::-1])
			plt.show()

	def __my_show_cam_on_image(self, img: list, mask: list, output_name='out/camcam.mp4'):
		time_length = len(mask)
		video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, self.origin_size)
		for i in range(time_length):
			heatmap = cv2.applyColorMap(np.uint8(255*mask[i]), cv2.COLORMAP_JET)
			heatmap = np.float32(heatmap)/255
			cam = heatmap+np.float32(img)
			cam = cam/np.max(cam)
			cam = np.uint8(255*cam)
			video.write(cam)
		video.release()


def main():
	args = TrainOptions('config/sync_multisensory.yaml').parse()
	model_ms = MultiSensory(sound_rate=16000, image_fps=25)
	grad_cam = GradCAM(model_ms, 'img_block0', (224, 224), 2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

	train_loader = LabDataLoader(args.train_list, args.batch_size,
	                             num_workers=args.num_workers,
	                             tgt_frame_num=args.tgt_frame_num,
	                             tgt_fps=args.tgt_fps,
	                             resolution=args.img_size,
	                             wav_hz=16000,
	                             avspeech_flag=args.tmp_flag,
	                             is_train=True, )
	valid_loader = LabDataLoader(args.val_list, args.batch_size,
	                             num_workers=args.num_workers,
	                             tgt_frame_num=args.tgt_frame_num,
	                             tgt_fps=args.tgt_fps,
	                             resolution=args.img_size,
	                             wav_hz=16000,
	                             avspeech_flag=args.tmp_flag,
	                             is_train=False, )

	for data in train_loader:
		a_img, a_wav_match = data
		grad_cam.my_forward(a_img, a_wav_match, label=(0, 1), output_name='out/match.mp4')


# 调用函数
if __name__ == '__main__':
	main()

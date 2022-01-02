import cv2
from .datasets import letterbox
from .general import check_img_size, check_requirements, non_max_suppression_face, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from .plots import plot_one_box
from .torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
import torch


def dynamic_resize(shape, stride=32):
	max_size = max(shape[0], shape[1])
	if max_size%stride != 0:
		max_size = (int(max_size/stride)+1)*stride
	return max_size


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
	# Rescale coords (xyxy) from img1_shape to img0_shape
	if ratio_pad is None:  # calculate from img0_shape
		gain = min(img1_shape[0]/img0_shape[0], img1_shape[1]/img0_shape[1])  # gain  = old / new
		pad = (img1_shape[1]-img0_shape[1]*gain)/2, (img1_shape[0]-img0_shape[0]*gain)/2  # wh padding
	else:
		gain = ratio_pad[0][0]
		pad = ratio_pad[1]

	coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
	coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
	coords[:, :10] /= gain
	# clip_coords(coords, img0_shape)
	coords[:, 0].clamp_(0, img0_shape[1])  # x1
	coords[:, 1].clamp_(0, img0_shape[0])  # y1
	coords[:, 2].clamp_(0, img0_shape[1])  # x2
	coords[:, 3].clamp_(0, img0_shape[0])  # y2
	coords[:, 4].clamp_(0, img0_shape[1])  # x3
	coords[:, 5].clamp_(0, img0_shape[0])  # y3
	coords[:, 6].clamp_(0, img0_shape[1])  # x4
	coords[:, 7].clamp_(0, img0_shape[0])  # y4
	coords[:, 8].clamp_(0, img0_shape[1])  # x5
	coords[:, 9].clamp_(0, img0_shape[0])  # y5
	return coords


def show_results(img, xywh, conf, landmarks, class_num):
	h, w, c = img.shape
	tl = 1 or round(0.002*(h+w)/2)+1  # line/font thickness
	x1 = int(xywh[0]*w-0.5*xywh[2]*w)
	y1 = int(xywh[1]*h-0.5*xywh[3]*h)
	x2 = int(xywh[0]*w+0.5*xywh[2]*w)
	y2 = int(xywh[1]*h+0.5*xywh[3]*h)
	cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

	clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

	for i in range(5):
		point_x = int(landmarks[2*i]*w)
		point_y = int(landmarks[2*i+1]*h)
		cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

	tf = max(tl-1, 1)  # font thickness
	label = str(int(class_num))+': '+str(conf)[:5]
	cv2.putText(img, label, (x1, y1-2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
	return img


# 输入人脸检测模型和frame, 输出边界框坐标
def face_detect(model_yolo, img: torch.Tensor):
	# img = (b, 3, w, h)
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	w = img.shape[2]
	h = img.shape[3]
	# Inference
	pred = model_yolo(img)[0]
	# pred = torch.Size([29, 4032, 16])
	# Apply NMS
	pred = non_max_suppression_face(pred, 0.5, 0.3)
	# pred = [29, [n, 16]]
	# 输入一共29张图片，[n,16]表示，第i张图片，检测到 n张人脸，16里的前 4个是x1,y1,x2,y2
	bbox_seq = []
	for i in range(len(pred)):
		# pred是否按照置信度排序了？需不需要计算图片大小做一次筛选？
		if pred[i] is None or len(pred[i])==0:
			bbox_seq.append([0, 0, w-1, h-1])
			# # 如果找不到人脸，就选择将整张图片、0置信度作为bbox返回
			# print('\nERROR: no face found, appending whole image\n')
		else:
			bbox_seq.append(pred[i][0, :4].cpu().detach().int().tolist())
	return bbox_seq

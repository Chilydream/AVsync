import torch
import torchvision.transforms as transforms
from utils.tensor_utils import PadSquare
from third_party.yolo.yolo_utils.util_yolo import face_detect


def crop_face_batch_seq(model_yolo, img_batch, face_size, run_device='cuda:0'):
	if isinstance(face_size, int):
		face_size = (face_size, face_size)
	if isinstance(run_device, str):
		run_device = torch.device(run_device)

	img_batch = img_batch.to(run_device)

	face_seq_list = []
	for img_seq in img_batch:
		face_seq = crop_face_seq(model_yolo, img_seq, face_size, run_device)
		face_seq_list.append(face_seq)
	face_batch = torch.stack(face_seq_list).to(run_device)
	# (b, seq, 3, res, res)
	return face_batch


def crop_face_seq(model_yolo, img_seq, face_size, run_device='cuda'):
	if isinstance(face_size, int):
		face_size = (face_size, face_size)
	if isinstance(run_device, str):
		run_device = torch.device(run_device)

	pad_resize = transforms.Compose([PadSquare(),
	                                 transforms.Resize(face_size)])

	img_seq = img_seq.to(run_device)
	with torch.no_grad():
		bbox_list = face_detect(model_yolo, img_seq)

	face_list = []
	for i, bbox in enumerate(bbox_list):
		x1, y1, x2, y2 = bbox
		xscale = max(0.05*(x2-x1), 2)
		yscale = max(0.05*(y2-y1), 2)
		x1 = int(max(0, x1-xscale))
		y1 = int(max(0, y1-yscale))
		x2 = int(min(img_seq.shape[3], x2+xscale))
		y2 = int(min(img_seq.shape[2], y2+yscale))
		face_list.append(pad_resize(img_seq[i, :, y1:y2, x1:x2]))
	face_seq = torch.stack(face_list).to(run_device)
	# (seq, 3, face_size, face_size)
	return face_seq

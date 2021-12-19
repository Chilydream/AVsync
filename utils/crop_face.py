import torch
import torchvision.transforms as transforms
from utils.tensor_utils import PadSquare
from third_party.yolo.yolo_utils.util_yolo import face_detect


def crop_face(model_yolo, img_batch, args):
	run_device = torch.device('cuda' if args.gpu else 'cpu')
	img_batch = img_batch.to(run_device)
	pad_resize = transforms.Compose([PadSquare(),
	                                 transforms.Resize(args.face_resolution)])

	with torch.no_grad():
		face_seq_list = []
		for img_seq in img_batch:
			face_list = []
			bbox_list = face_detect(model_yolo, img_seq)
			for i, bbox in enumerate(bbox_list):
				x1, y1, x2, y2 = bbox
				if x1>=x2:
					x1, x2 = 0, args.img_resolution-1
				if y1>=y2:
					y1, y2 = 0, args.img_resolution-1
				face_list.append(pad_resize(img_seq[i, :, y1:y2, x1:x2]))
			face_seq = torch.stack(face_list).to(run_device)
			face_seq_list.append(face_seq)
		face_batch = torch.stack(face_seq_list).to(run_device)
		# (b, seq, 3, res, res)
		return face_batch

import torch

from third_party.HRNet.utils_inference import get_batch_lmks
from utils.GetDataFromFile import get_frame_tensor


def extract_lmk(model_hrnet, mp4_name, run_device):
	lmk_name = mp4_name[:-3]+'lmk'
	img_seq = get_frame_tensor(mp4_name)
	img_seq = img_seq.to(run_device)
	# img_seq = (29, 3, 256, 256)
	lmk_seq = get_batch_lmks(model_hrnet, img_seq)
	# lmk_seq = (29, 68, 2)
	torch.save(lmk_seq, lmk_name)

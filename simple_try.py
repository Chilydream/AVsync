import multiprocessing as mp
import torch
import sys

sys.path.append('./third_party/yolo')
from utils.GetDataFromFile import get_frame_tensor
from utils.crop_face import crop_face_seq
from third_party.yolo.yolo_models.yolo import Model as yolo_model


def func1(idx):
	model_yolo = yolo_model(cfg='config/yolov5s.yaml').float().fuse().eval()
	model_yolo.load_state_dict(torch.load('pretrain_model/raw_yolov5s.pt'))
	run_device = torch.device('cuda:0')
	model_yolo = model_yolo.to(run_device)
	with open('metadata/LRW_train_3090.txt', 'r') as fr, open(f'log/face{idx}.log', 'w') as fw:
		lines = fr.readlines()
		for i, line in enumerate(lines):
			if i%8!=idx:
				continue
			word, mp4name = line.strip().split('\t')
			facename = mp4name[:-3]+'face'
			img_seq = get_frame_tensor(mp4name)
			face_tensor = crop_face_seq(model_yolo, img_seq, 160, run_device)
			torch.save(face_tensor, facename)
			print(f'{word}\t{facename}', file=fw)
			if (i-idx)%1000==0:
				print(f'{idx} thread finish {i}')



def main():
	mp.set_start_method('spawn')
	process_list = []
	for i in range(8):
		process_list.append(mp.Process(target=func1, args=(i,)))
	[p.start() for p in process_list]
	[p.join() for p in process_list]
	print(f'main thread end')


if __name__ == '__main__':
	main()

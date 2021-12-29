import os
import cv2
import glob
import tqdm
from utils.extract_wav import extract_wav


def get_LRW_meta(metafile, mode):
	assert mode in ('train', 'val', 'test')
	dataset_dir = '/home/tliu/fsx/dataset/LRW'
	word_list = []
	for filename in os.listdir(dataset_dir):
		if os.path.isdir(os.path.join(dataset_dir, filename)):
			word_list.append(filename)
	word_list.sort()
	with open(metafile, 'w') as fw:
		for word in tqdm.tqdm(word_list):
			end_dir = os.path.join(dataset_dir, word, mode)
			video_list = glob.glob(os.path.join(end_dir, '*.mp4'))
			video_list.sort()
			for filename in video_list:
				new_filename = filename.replace("\\", "/")
				wavname = new_filename.replace('mp4', 'wav')
				if not os.path.exists(wavname):
					print(f'{wavname} not exists, extracting')
					extract_wav(filename)
				print(f'{word}\t{new_filename}', file=fw)


def get_Lab_meta(metafile, mode):
	assert mode in ('train', 'val', 'test')
	dataset_dir = '/data1/lab_regular'
	silent_dir = os.path.join(dataset_dir, 'silent')
	speak_dir = os.path.join(dataset_dir, 'talk')
	with open(metafile, 'w') as fw:
		video_list = glob.glob(os.path.join(silent_dir, '*.mp4'))
		video_list.sort()
		for i, filename in enumerate(video_list):
			new_filename = filename.replace('\\', '/')
			if i%10<=7 and mode=='train':
				print(f'0\t{new_filename}', file=fw)
			elif i%10==8 and mode=='val':
				print(f'0\t{new_filename}', file=fw)
			elif i%10==9 and mode=='test':
				print(f'0\t{new_filename}', file=fw)
		video_list = glob.glob(os.path.join(speak_dir, '*.mp4'))
		video_list.sort()
		for i, filename in enumerate(video_list):
			new_filename = filename.replace('\\', '/')
			if i%10<=7 and mode=='train':
				print(f'1\t{new_filename}', file=fw)
			elif i%10==8 and mode=='val':
				print(f'1\t{new_filename}', file=fw)
			elif i%10==9 and mode=='test':
				print(f'1\t{new_filename}', file=fw)


def word_split():
	old_train = 'metadata/LRW_train_3090.txt'
	old_val = 'metadata/LRW_val_3090.txt'
	old_test = 'metadata/LRW_test_3090.txt'
	new_train = 'metadata/LRW_train_3090.txt'
	new_val = 'metadata/LRW_val_3090.txt'
	new_test = 'metadata/LRW_test_3090.txt'


if __name__ == '__main__':
	get_LRW_meta('metadata/LRW_train_3090.txt', 'train')
	get_LRW_meta('metadata/LRW_val_3090.txt', 'val')
	get_LRW_meta('metadata/LRW_test_3090.txt', 'test')

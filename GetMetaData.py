import os
import cv2
import glob
import tqdm

from utils.GetDataFromFile import get_video_time
from utils.extract_wav import extract_wav


def get_LRW_meta(metafile, mode):
	# metafile = 'metadata/LRW_test_3090.txt'
	# mode = 'test'
	assert mode in ('train', 'val', 'test')
	dataset_dir = '/hdd1/dataset/LRW'
	word_list = []
	for filename in os.listdir(dataset_dir):
		if os.path.isdir(os.path.join(dataset_dir, filename)):
			word_list.append(filename)
	word_list.sort()
	with open(metafile, 'w') as fw:
		for idx, word in enumerate(word_list):
			if idx%10<8 and mode!='train':
				continue
			elif idx%10==8 and mode!='val':
				continue
			elif idx%10==9 and mode!='test':
				continue
			end_dir = os.path.join(dataset_dir, word, 'train')
			face_list = glob.glob(os.path.join(end_dir, '*.face'))
			face_list.sort()
			for filename in face_list:
				new_filename = filename.replace("\\", "/")
				mp4name = new_filename.replace('/hdd1', '/home/tliu/fsx').replace('face', 'mp4')
				print(f'{word}\t{mp4name}', file=fw)


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
			if i%10<=7 and mode == 'train':
				print(f'0\t{new_filename}', file=fw)
			elif i%10 == 8 and mode == 'val':
				print(f'0\t{new_filename}', file=fw)
			elif i%10 == 9 and mode == 'test':
				print(f'0\t{new_filename}', file=fw)
		video_list = glob.glob(os.path.join(speak_dir, '*.mp4'))
		video_list.sort()
		for i, filename in enumerate(video_list):
			new_filename = filename.replace('\\', '/')
			if i%10<=7 and mode == 'train':
				print(f'1\t{new_filename}', file=fw)
			elif i%10 == 8 and mode == 'val':
				print(f'1\t{new_filename}', file=fw)
			elif i%10 == 9 and mode == 'test':
				print(f'1\t{new_filename}', file=fw)


def word_split():
	old_train = 'metadata/LRW_train_3090.txt'
	split_train = 'metadata/LRW_train_3090_train.txt'
	split_val = 'metadata/LRW_train_3090_val.txt'
	split_test = 'metadata/LRW_train_3090_test.txt'

	char2word = dict()
	with open(old_train, 'r') as fr:
		lines = fr.readlines()
		for line in lines:
			word, filename = line.strip().split('\t')
			if len(word) == 1:
				print(word, filename)
			if word[0] not in char2word.keys():
				char2word[word[0]] = set()
			char2word[word[0]].add(word)
		val_word_list = set()
		test_word_list = set()
		for c_iter, word_list in char2word.items():
			word_num = len(word_list)
			tmp_list = list(word_list)
			tmp_list.sort()
			val_word_list.add(tmp_list[int(0.3*word_num)])
			val_word_list.add(tmp_list[int(0.7*word_num)])
			test_word_list.add(tmp_list[int(0.5*word_num)])
		test_word_list = test_word_list-val_word_list

		with open(split_train, 'w') as strain, open(split_val, 'w') as sval, open(split_test, 'w') as stest:
			for line in lines:
				word, filename = line.strip().split('\t')
				if word in val_word_list:
					print(f'{word}\t{filename}', file=sval)
				elif word in test_word_list:
					print(f'{word}\t{filename}', file=stest)
				else:
					print(f'{word}\t{filename}', file=strain)


def get_LRS2_meta(metafile, mode):
	assert mode in ('train', 'val', 'test')
	dataset_dir = '/home/tliu/fsx/dataset/LRS2'
	main_dir = os.path.join(dataset_dir, 'main')
	pretrain_dir = os.path.join(dataset_dir, 'pretrain')
	with open(metafile, 'w') as fw:
		video_list = glob.glob(os.path.join(main_dir, '*', '*.mp4'))
		# video_list.extend(glob.glob(os.path.join(pretrain_dir, '*', '*.mp4')))
		video_list.sort()
		for i, filename in tqdm.tqdm(enumerate(video_list)):
			new_filename = filename.replace('\\', '/')
			if i%10<=7 and mode == 'train':
				video_time = get_video_time(new_filename)
				print(f'1\t{new_filename}\t{video_time}', file=fw)
			elif i%10 == 8 and mode == 'val':
				video_time = get_video_time(new_filename)
				print(f'1\t{new_filename}\t{video_time}', file=fw)
			elif i%10 == 9 and mode == 'test':
				video_time = get_video_time(new_filename)
				print(f'1\t{new_filename}\t{video_time}', file=fw)

if __name__ == '__main__':
	get_LRS2_meta('./metadata/LRS2_train.txt', 'train')
	get_LRS2_meta('./metadata/LRS2_val.txt', 'val')
	get_LRS2_meta('./metadata/LRS2_test.txt', 'test')

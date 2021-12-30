import os
import cv2
import glob
import tqdm
from utils.extract_wav import extract_wav


def get_LRW_meta(metafile, mode):
	# metafile = 'metadata/LRW_test_3090.txt'
	# mode = 'test'
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


if __name__ == '__main__':
	word_split()

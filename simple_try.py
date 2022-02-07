import os
import glob
import shutil

data_dir = '/home/tliu/fsx/dataset/avspeech'
mp4list = glob.glob(os.path.join(data_dir, '*.mp4'))
print(len(mp4list))

ftrain = open('metadata/avspeech_train.txt', 'w')
fval = open('metadata/avspeech_val.txt', 'w')
ftest = open('metadata/avspeech_test.txt', 'w')

for idx, mp4name in enumerate(mp4list):
	if idx%10<8:
		fw = ftrain
	elif idx%10==8:
		fw = fval
	else:
		fw = ftest
	print(mp4name, file=fw)

import os
import glob
import shutil
import cv2
import numpy as np
import scipy.io.wavfile as wavf
from utils.GetDataFromFile import get_frame_and_wav_cv2
from utils.extract_wav import extract_wav

time_length = 7
origin_time = 50
frag = origin_time/(time_length-1)
final_list = []
j = 0
for i in range(time_length-1):
	while j<=frag*(i+1):
		left_dist = (j-frag*i)/frag
		final_list.append(i+left_dist)
		j += 1

print(final_list)
print(len(final_list))

import os
from ffmpy3 import FFmpeg


def extract_wav(mp4name, ar=None):
	wavname = mp4name[:-3]+'wav'
	# if not os.path.exists(wavname):
	# 	if ar is None:
	# 		gopt = []
	# 	else:
	# 		gopt = [f'-ar {ar}']
	# 	ff = FFmpeg(inputs={filename: None},
	# 	            global_options=gopt,
	# 	            outputs={wavname: '-vn -f wav'})
	# 	ff.run()
	os.system(f'ffmpeg -i {mp4name} -f wav -ac 1 -ar 16000 {wavname}')



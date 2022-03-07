import os
from ffmpy3 import FFmpeg


def extract_wav(mp4name):
	wavname = mp4name[:-3]+'wav'
	if not os.path.exists(wavname):
		# ff = FFmpeg(inputs={mp4name: None},
		#             outputs={wavname: '-vn -f wav'})
		# ff.run()
		os.system(f'ffmpeg -i {mp4name} -f wav -ac 1 -ar 16000 {wavname} -loglevel quiet')
	else:
		print('wav file already exists')



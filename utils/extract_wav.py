import os
from ffmpy3 import FFmpeg


def extract_wav(filename):
	wavname = filename[:-3]+'wav'
	if not os.path.exists(wavname):
		ff = FFmpeg(inputs={filename: None},
		            outputs={wavname: '-vn -f wav'})
		ff.run()

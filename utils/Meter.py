class Meter(object):
	def __init__(self, name, display, fmt=':f', end=', ', beta_ema=0.95):
		self.name = name
		self.display = display
		self.fmt = fmt
		self.start_time = 0
		self.end = end
		self.beta_ema = beta_ema

		self.val, self.avg, self.sum = None, None, None
		self.avg_ema = None
		self.count, self.start_time, self.time = None, None, None
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.avg_ema = 0
		# exponential moving average
		self.sum = 0
		self.count = 0
		self.start_time = 0
		self.time = 0

	def set_start_time(self, start_time):
		self.start_time = start_time

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg_ema = self.beta_ema*self.avg_ema+(1-self.beta_ema)*val
		self.avg = self.sum/self.count
		self.time = val-self.start_time

	def __str__(self):
		fmtstr = '{name}:{'+self.display+self.fmt+'}'+self.end
		return fmtstr.format(**self.__dict__)

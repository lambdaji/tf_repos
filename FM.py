#coding=utf8

class BernoulliArm(object):
	"""docstring for BernoulliArm"""
	def __init__(self, p):
		self.p = p
		
	def draw(self):
		if random.random() > self.p:
			return 0.0
		else:
			return 1.0

def main():
	means = [0.1, 0.1, 0.1, 0.1, 0.9]
	n_arms = len(means)
	random.shuffle(means)
	arms = map(lambda mu: BernoulliArm(mu), means)
	


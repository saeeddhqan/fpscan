import random, numpy, math, torch
nn = torch.nn


def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)


b, d, l = 1, 4, 16
x = torch.rand(b, d, l)
conv1 = nn.Conv1d(d, 1, 4)
print(conv1.weight.shape)
print(conv1.bias)
print(conv1(x).shape)
print(conv1(x)[0])
print(torch.sum(conv1.weight * x[:,:,:4]))
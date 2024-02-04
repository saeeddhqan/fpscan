
import torch
import random, math, numpy

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)

# Define the forward operation
def naive_pscan(A, X, Y_init):
	y = Y_init
	s = 0
	o = []
	for k in range(A.size(1)):
		y = A[:, k] * y + X[:, k]
		s = s + y
		o.append(y)
	o = torch.stack(o, dim=1)
	return o

# Define the loss function
def loss_function(o, target):
	return torch.sum((o - target) ** 2)  # Example loss function, could be anything

# Example input tensors
A = torch.randn(3, 5, requires_grad=True)
X = torch.randn(3, 5, requires_grad=True)
Y_init = torch.randn(3, requires_grad=True)

# Forward pass
o = naive_pscan(A, X, Y_init)

# Example target tensor
target = torch.randn_like(o)

# Compute loss
loss = loss_function(o, target)

# Backward pass to get gradients using PyTorch autograd
loss.backward()

# Access gradients
grad_A_autograd = A.grad
grad_X_autograd = X.grad
grad_Y_init_autograd = Y_init.grad

# Reset gradients for future computations
A.grad = None
X.grad = None
Y_init.grad = None

# Manual gradient calculations
grad_loss_o = 2 * (o - target)
grad_A_manual = torch.zeros_like(A)
grad_X_manual = torch.zeros_like(X)
grad_Y_init_manual = torch.zeros_like(Y_init)

for i in range(A.size(0)):
	for j in range(A.size(1)):
		grad_A_manual[i, j] = torch.sum(grad_loss_o[:, j] * Y_init)
		grad_X_manual[i, j] = torch.sum(grad_loss_o[:, j])
		grad_Y_init_manual += torch.sum(grad_loss_o[:, j] * A[:, j])

# Compare gradients
print("Gradient of A (Autograd):", grad_A_autograd)
print("Gradient of A (Manual):", grad_A_manual)

print("Gradient of X (Autograd):", grad_X_autograd)
print("Gradient of X (Manual):", grad_X_manual)

print("Gradient of Y_init (Autograd):", grad_Y_init_autograd)
print("Gradient of Y_init (Manual):", grad_Y_init_manual)
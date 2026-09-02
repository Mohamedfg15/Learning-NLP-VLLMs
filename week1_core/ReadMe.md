# Day 1 — PyTorch & Autograd

## Topics

- Tensors
- Tensor Operations
- Autograd
- Computational Graph
- Gradients
- Manual Gradient Descent
- nn.Module
- nn.Parameter
- Loss Functions
- Optimizers
- Training Loop
- Validation

## Main Implementation

`train_from_scratch.py`

## Training Flow

Data
→ Forward Pass
→ Loss
→ Backward Pass
→ Gradients
→ Optimizer Step
→ Updated Parameters

## What I Practiced

- Built a model using `nn.Module`
- Used `nn.Parameter` / trainable parameters
- Implemented a training loop
- Used MSE loss
- Used SGD
- Implemented validation
- Logged training loss

## Key Interview Concepts

- `loss.backward()`
- `optimizer.zero_grad()`
- `optimizer.step()`
- Epoch vs Iteration
- Training vs Evaluation
- `torch.no_grad()`
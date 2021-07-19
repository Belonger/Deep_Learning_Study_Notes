import random
import torch

"""y = 5x"""

x = torch.normal(0, 1, (1000, 1))
true_w = torch.tensor([5.0])
y = torch.matmul(x, true_w)
y = y.reshape(x.shape)

# 先假设 w = 10
w = torch.tensor([10.0], requires_grad=True)
a = 0.5 # a为学习率，步长

for i in range(50):
    loss = ((w * x[i] - y[i]) ** 2) / 2
    loss.backward()
    with torch.no_grad():
        w -= a * w.grad
        w.grad.zero_()
        print(f'第 {i + 1} 轮，w为：{w}')

print(true_w, w)

# 这一次我们的目的是理解小批量(随机)梯度下降法

import random
import torch

"""这次仍然以y = 5x 为例"""

x = torch.normal(0, 1, (2000, 1))               # 形状为：1000行，1列
true_w = torch.tensor([5.0])
y = torch.matmul(x, true_w)
y = y.reshape(x.shape)

w = torch.tensor([10.0], requires_grad=True)
lr = 0.2

# 先不用小批量来算
# for i in range(1000):
#     loss = ((w * x[i] - y[i]) ** 2) / 2
#     loss.backward()
#     with torch.no_grad():
#         w -= lr * w.grad
#         w.grad.zero_()
#         print(f'第 {i + 1} 轮，预测w值为：{w}')
#
# print(true_w, w)

# 用小批量来算
# 先不写迭代函数
for i in range(0,500,10):
    loss = ((w * x[i:(i+10)] - y[i:(i+10)]) ** 2) / 2
    loss.sum().backward()
    with torch.no_grad():
        w -= lr * w.grad / 10
        w.grad.zero_()
        print(f'第 {i + 1} 轮，w的预测值为：{w}')

print(true_w, w)

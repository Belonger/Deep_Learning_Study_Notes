import random
import torch

# 我们就用y = wx 来理解

def synthetic_data(w, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:min(i+batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]

def linear(X, w):
    return torch.matmul(X, w)

def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2

def sgd(params, lr, batch_size):                    # 参数的更新
    with torch.no_grad():
        for param in params:
            param -= (lr * param.grad) / batch_size
            param.grad.zero_()

if __name__ == '__main__':
    '''y = 5x'''
    true_w = torch.tensor([100.0])
    features, labels = synthetic_data(true_w, 1000)
    w = torch.tensor([1.0], requires_grad=True)
    batch_size = 10
    lr = 0.01
    num_epochs = 100
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(linear(X, w), y)
            l.sum().backward()
            sgd([w], lr, batch_size)         # 参数更新
        with torch.no_grad():
            train_l = squared_loss(linear(features, w), labels)
            print(f' epoch {epoch + 1}, loss {train_l.mean()}')

    print(true_w, w)

















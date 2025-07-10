import sys, os
try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base = os.getcwd()
sys.path.append(base)

import numpy as np
import math
import matplotlib.pyplot as plt
from dezero import Variable, Function, as_array
import dezero.functions as F


## 선형회귀
x_data = np.random.rand(100,1)
y_data = 5 + 2*x_data + np.random.rand(100,1)
x, y = Variable(x_data), Variable(y_data)
W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x,W) + b
    return y

lr = 0.1
iters=100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

# 5-2. 회귀선 vs 데이터
plt.scatter(x_data, y_data, label="Data")
x_line = np.linspace(0, 1, 100).reshape(100, 1)
y_line = W.data * x_line + b.data
plt.plot(x_line, y_line, color='red', label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()
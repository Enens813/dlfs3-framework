import sys, os
try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base = os.getcwd()
sys.path.append(base)

import numpy as np
import math
from dezero import Variable, Function, as_array

## 고차미분 test
def f(x):
    y = x**4 -2 * x**2
    return y

# forward
x = Variable(np.array(2.0))
y = f(x)

# 1차미분. y' = 4 * x**3 - 4 * x -> 24
y.backward(create_graph=True)
print(x.grad)

# 2차미분 y'' = 12 * x**2 - 4 -> 44
gx = x.grad
x.cleargrad()   # 미분값 재설정
gx.backward()
print(x.grad)
print()



## newton's method
x = Variable(np.array(2.0))
iters=10
for i in range(iters):
    print(i,x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data



## sin함수의 고차미분
import matplotlib.pyplot as plt
import dezero.functions as F

x = Variable(np.linspace(-7,7,200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]
for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)


labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()


## 고차미분 과정 계산 그래프 그리기
from dezero.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name='x'
y.name='y'
y.backward(create_graph=True)

iters=1 # 2차미분
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters+i)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')


## 미분방정식 풀기
x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph = True)
gx = gx.grad
x.cleargrad()
z = gx**3 + y
z.backward()
print(x.grad)   # 정답: 100
import sys, os
try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base = os.getcwd()
sys.path.append(base)

import numpy as np
import math
from dezero import Variable, Function, as_array

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):     # taylor 급수로 점점 차수를 늘려가며 더하는 값이 threshold 보다 작아질 때까지 계산
        c = (-1) **i / math.factorial(2*i + 1)  
        t = c * x ** (2*i + 1)
        y = y+t
        if abs(t.data) < threshold:
            break
    return y

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 **2) **2 + (1 - x0) **2
    return y

if __name__ == '__main__':
    x0 = Variable(as_array(0.0))
    x1 = Variable(as_array(2.0))
    y = rosenbrock(x0, x1)
    y.backward()
    print(x0.grad, x1.grad)

    ## gradient descent
    lr = 0.001
    iters = 1000

    # 경로 저장용
    x0_path = []
    x1_path = []

    for i in range(iters):
        x0_path.append(x0.data.copy())
        x1_path.append(x1.data.copy())
        print(x0, x1)
        
        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad
    
    # 시각화 용
    x0_vals = np.linspace(-2, 2, 100)
    x1_vals = np.linspace(-1, 3, 100)
    X0, X1 = np.meshgrid(x0_vals, x1_vals)
    Z = rosenbrock(X0, X1)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.contour(X0, X1, Z, levels=50, cmap='gray')  # 등고선

    # 경로 그리기
    plt.plot(x0_path, x1_path, 'ro-', markersize=3, label='gradient path')

    # 최솟값 지점 표시 (x0=1, x1=1)
    plt.plot(1, 1, 'b*', markersize=12, label='minimum')

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('Gradient Descent on Rosenbrock Function')
    plt.legend()
    plt.grid(True)
    plt.show()

    ## Newton's method (lr = 1/f'')
    def f(x): return x**4 - 2 * x**2
    def gx2(x): return 12 * x**2 - 4    # 수기로 2차미분

    x = Variable(np.array(2.0))
    iters=10

    x_list = []
    y_list = []
    for i in range(iters):
        print(i, x)
        y = f(x)
        x_list.append(x.data.copy())
        y_list.append(y.data.copy())

        x.cleargrad()
        y.backward()
        x.data -= x.grad / gx2(x.data)
    
    xs = np.linspace(-2.5, 2.5, 300)
    ys = f(xs)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label='$f(x) = x^4 - 2x^2$', color='gray')

    # 경로 시각화
    plt.plot(x_list, y_list, 'ro-', label='Newton steps')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Newton\'s Method for $f(x) = x^4 - 2x^2$')
    plt.grid(True)
    plt.legend()
    plt.show()
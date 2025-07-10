import sys, os
try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base = os.getcwd()
sys.path.append(base)

import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)    ## np.cos 아님!!
        return gx
    
def sin(x):
    return Sin()(x)



class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)       ## np.sin 아님!!
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):     # y=tanh 일 때, y' = 1-y^2 (tanh(x) = (e^x - e^-x)/(e^x + e^-x) 미분해보면 나옴)
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)


"""
Variable에 다차원 np.array를 대입하면, 각각을 scalar(0차원 텐서)로 취급하고 각 scalar에 대해 계산.
matrix multiplication이 있어도, 대각성분들이 모두 독립이기 때문에 0. -> scalar로 취급해서 계산할 때와 같음.
즉, 지금까지의 forward, backward 계산은 문제 없음
아래는 scalar 취급하지 않는 함수들을 작성.
"""

from dezero.core import as_variable

class Reshape(Function):
    def __init__(self, shape):  # 목표가 되는 형상
        self.shape = shape
        
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        gx = np.transpose(gy)
        return gx
    
def transpose(x):
    return Transpose()(x)

if __name__ == '__main__':
    from dezero.core import Variable, setup_variable
    
    setup_variable()

    x = Variable(np.random.randn(2,3))
    y = x.reshape(3,2)
    z = y.T
    print(x)
    print(y)
    print(z)
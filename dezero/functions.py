import sys, os
try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base = os.getcwd()
sys.path.append(base)

import numpy as np
from dezero.core import Function
from dezero import utils

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


class Sum(Function):
    def __init__ (self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) # numpy 문제로, gy를 조금 바꿔줌. 자세한 설명은 utils에서
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

# x=[1,2,3] -> y=np.broadcast_to(x,(2,3)) 하면 y= [[1,2,3], [1,2,3]] 이 됨. 즉 확장해줌. 대신, gy = [[2,3,4], [2,3,4]]라면 gx = [4,6,8] 이 됨. 역변환은 sum_to로 정의함
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)   # utils에서 정의
        return gx

def broadcast_to(x, shape):
    if x.shape ==shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape ==shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)        # np.dot(x,W)보다 ndarray 인스턴스에도 대응할 수 있음
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x,W):
    return MatMul()(x, W)


if __name__ == '__main__':
    from dezero.core import Variable, setup_variable
    
    setup_variable()

    x = Variable(np.random.randn(2,3))
    y = x.reshape(3,2)
    z = y.T
    print(x)
    print(y)
    print(z)
    print()

    ## broadcast 확인
    x0 = Variable(np.array([1,2,3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    print(x0, x1, y)
    y.backward()
    print(x1.grad)

    ## matmul 확인
    x = Variable(np.random.randn(2,3))
    W = Variable(np.random.randn(3,4))
    y = matmul(x,W)
    y.backward()
    print(x.shape, W.shape)
    print(x.grad.shape, W.grad.shape)
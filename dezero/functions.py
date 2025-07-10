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
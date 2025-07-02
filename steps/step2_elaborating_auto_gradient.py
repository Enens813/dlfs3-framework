import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은(는) 지원하지 않습니다".format(type(data)))
            
        self.data = data
        self.grad = None    
        self.creator = None 
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)  

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x): 
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # *(list unpack)을 이용해서 가변길이 input/output
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys ]

        for output in outputs:
            output.set_creator(self)    

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs)>1 else outputs[0] # output길이가 2 이상이면 array of Variable로, 1이면 Variable로 반환
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy): 
        x = self.input.data
        gx = 2 * x * gy 
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx = (gy, gy)
        return gx

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    return Add()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)


if __name__ == '__main__':
    
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))

    y = add(x0, x1)
    print(y.data)
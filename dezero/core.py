"""
고차미분이 적용된 core_simple
y=sin(x)의 역전파 식을 구할 때, dy/dx = cos(x) 이므로, dL/dx = dL/dy * dy/dx -> gx = gy * cos(x)
gx.backward()를 할 수 있다면 d(gx)/dx = d(dL/dx)/dx 이 구해지는 것이므로 2차미분 및 전파 가능. 

"""
import numpy as np
import heapq
import itertools
import weakref
import contextlib

## 설정 관련
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)       
    setattr(Config, name, value)            
    try:                                    
        yield
    finally:                                
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

## format 변환 관련
def as_array(x): 
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


## forward, backward 실행 관련
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은(는) 지원하지 않습니다".format(type(data)))    
            
        self.data = data
        self.name = name
        self.grad = None    
        self.creator = None
        self.generation = 0     
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 
    
    def backward(self, retain_grad=False, create_graph=False): 
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()
        counter = itertools.count()  

        def add_func(f):    
            if f not in seen_set:
                seen_set.add(f)
                heapq.heappush(funcs, (-f.generation, next(counter), f)) 
        
        add_func(self.creator)

        while funcs:
            _, _, f  = heapq.heappop(funcs)  

            gys = [output().grad for output in f.outputs] 

            # create_graph = False일 때 역전파를 아예 하지 않게 설정
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)  
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):    
                    if x.grad is None:              
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None     
    
    def cleargrad(self):
        self.grad = None
    
    @property 
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):         
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+ ' '*9)
        return f'variable({p})'


class Function:
    def __call__(self, *inputs): 
        inputs = [as_variable(x) for x in inputs]

        # 순전파 계산
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys ]

        # '연결' 생성
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   

            for output in outputs:
                output.set_creator(self)    

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs)>1 else outputs[0] 
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()


## 연산자 오버로딩
# Add는 안바뀜
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return (gy, gy)
    
# Mul은 backward 시, data로 받는게 아니라 variable 그 자체로 받도록 수정
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy*x1, gy*x0

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    def backward(self, gy):
        return gy, -gy
    
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1 **2))
        return gx0, gx1

class Pow(Function):
    def __init__ (self, c):
        self.c = c
    def forward(self, x):
        y = x ** self.c
        return y
    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * self.c * ( x ** (self.c-1))
        return gx
    

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x): return Neg()(x)

def sub(x0, x1): 
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x,c): return Pow(c)(x)

def setup_variable():    
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
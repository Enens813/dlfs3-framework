import numpy as np
import heapq
import itertools


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은(는) 지원하지 않습니다".format(type(data)))
            
        self.data = data
        self.grad = None    
        self.creator = None
        self.generation = 0     # 복잡한 계산 그래프에서 backprop순서 지정을 하기 위해서 필요
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        counter = itertools.count()  # 고유 번호 생성

        def add_func(f):    # 지금까지 나온 적이 없으면 funcs 라는 heapq(min-heap(항상 root node에 최솟값) 자료구조)에 추가. min-heap에서 최솟값을 판단하는 기준(priority)는 -generation으로 함. generation이 큰 게 root가 되도록.
            if f not in seen_set:
                seen_set.add(f)
                heapq.heappush(funcs, (-f.generation, next(counter), f)) # f.generation과 f가 같을 경우, 임의 순서대로 진행되도록 counter 추가
        
        add_func(self.creator)

        while funcs:
            # funcs.sort(key=lambda x: x.generation)
            # f = funcs.pop()
            _, _, f  = heapq.heappop(funcs)  # 가장 큰 generation을 pop (위 두 줄을 우선순위 큐로 구현)

            gys = [output.grad for output in f.outputs] # output 여러개가 된 걸 적용
            gxs = f.backward(*gys)  # unpack해서 backward 에 넣어줌 (list가 아니라 array 여러개로)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):    # 가변길이 적용, 각 input마다 grad 입력해줌
                if x.grad is None:              # add(x,x) 처럼 한 변수가 두번 들어가는 경우를 위해 if~else 추가
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                    # x.grad += gx 로 하면 안됨. 
                    # x+=x는 in-place operation(덮어쓰기, overwrite), x=x+x는 값을 복사하여 새로 생성하는 것.
                    # backprop할 때 z=add(add(x,x),x) 하면, outputs=[z], gys=[z.grad], gxs=(z.grad,z.grad), 첫for step에서 x.grad=z.grad(같은 메모리 참조), 두번째 for step에서 x.grad += z.grad 하면, 메모리 상에선 a+=a인 것이므로 a<-2a와 같은 결과가 되어 z.grad=x.grad=2a가 되는 것
                    # 이게 python float에선 아닌데, ndarray에선 성립
                if x.creator is not None:
                    add_func(x.creator)
    
    def cleargrad(self):
        self.grad = None

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

        self.generation = max([x.generation for x in inputs])   # input의 generation 중 max로 함. 다음 변수의 generation을 지정할 때 사용

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
        x = self.inputs[0].data # 가변길이 input 형식 적용
        gx = 2 * x * gy 
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return (gy, gy)

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

    y = add(square(x0), square(x1))
    y.backward()
    print(f"y = x0^2 + x1^2, x0={x0.data}, x1={x1.data} : y, x0.grad, x1.grad")
    print(y.data)
    print(x0.grad)
    print(x1.grad)

    x0.cleargrad()
    z = add(add(x0, x0), x0)
    z.backward()
    print(f"z = x0 + x0 + x0, x0={x0.data} : z, z.grad, x0.grad")
    print(z.data)
    print(z.grad, id(z.grad))
    print(x0.grad, id(x0.grad)) # 3여야 함!!

    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(f"a = x^2, y = a^2 + a^2, x={x.data} : y, x.grad")
    print(y.data)
    print(x.grad)
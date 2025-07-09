import numpy as np
import heapq
import itertools
import weakref
import contextlib

# Config는 하나만 있는 게 좋음. -> config class는 인스턴스화 하지 않고 클래스 상태로 이용
# 전역 설정을 위한 static 변수라고도 부름
class Config:
    enable_backprop = True

@contextlib.contextmanager  # with using_config(): 처럼 쓸 수 있음
def using_config(name, value):
    old_value = getattr(Config, name)       # 설정을 바꾸기 전의 값(Config.name)을 저장
    setattr(Config, name, value)            # Config.name을 valeu로 바꿈
    try:                                    # with : 내부의 코드 실행
        yield
    finally:                                # with 내부 코드 실행 후 설정을 원래대로 복원
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은(는) 지원하지 않습니다".format(type(data)))
            
        self.data = data
        self.name = name
        self.grad = None    
        self.creator = None
        self.generation = 0     # 복잡한 계산 그래프에서 backprop순서 지정을 하기 위해서 필요
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 
    
    def backward(self, retain_grad=False): # retain_grad=False 일 때는 backprop 후 메모리를 해제함 (주로 backprop은 말단 변수의 grad만 필요하므로)
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

            gys = [output().grad for output in f.outputs] # output 여러개가 된 걸 적용      # function의 outputs를 weakref하면서 각 output 대신 output() 사용해야 함
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
            
            # backprop할 때 중간 단계의 grad는 전파하고 나서 지움. (backprop의 output은 (f의 input))
            # 왜냐면 우리는 dy/dy 같은걸 구하려는 게 아니고 dy/dW, dy/dx를 구하려는 것이기 때문. 필요없는 단계들이 좀 있음. 때문에 중간단계인 계산과정은 필요가 없음
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None     # y는 weakref라 () 써야 함
    
    def cleargrad(self):
        self.grad = None
    
    @property # shape method를 인스턴스 "변수"처럼 사용할 수 있음 x.shape() 대신 x.shape로 호출 가능
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
    
    def __repr__(self):         # 원래는 print(x)했을 때 <__main__.Variable object at 0x0000016B516DA450> 처럼 뜨던 걸 바꿔줌
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+ ' '*9)
        return f'variable({p})'

    

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

        # enable_backprop = True일 때만 backprop 코드 실행
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])   # input의 generation 중 max로 함. 다음 변수의 generation을 지정할 때 사용

            for output in outputs:
                output.set_creator(self)    

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            ## 파이썬의 메모리 관리 방식 
            # 참조 카운트: 파이썬에서는 메모리의 참조 수를 세는 '참조 카운트'방식과, GC(Garbage Collector) 방식을 모두 이용함. 
            # 순환 참조: a가 b를 참조하고, b가 a를 참조하는 상황은, 외부에서 a나 b를 참조하지 않아도 참조 카운트가 각각 1로 유지됨 -> 메모리에서 사라지지 않음.
            # 이때 참조 카운트 방식은 순환 참조를 해결할 수 없고 GC는 느림 -> 일단 순환 참조를 해결하면 빨라짐(메모리 누수 방지 즉, 쓸모없는 변수에 메모리를 할당하지 않음 & GC 오버헤드 감소 & del 연산 속도 증가).

            ## variable과 function의 순환참조
            # 위 방식에는 function -(output)-> variable, function <-(creator)- variable 즉 Variable.creator.outputs = [Variable] 의 순환참조 구조가 존재
            # retain_grad=False면 중간 단계의 grad를 유지하지 않기 위해 수동으로 y.grad=None을 하여 메모리 관리를 하려고 함
            # for y in f.outputs: y.grad=None을 하면 None이 되는 것처럼 보임. (순환 참조가 없다면 변수에 None을 대입했을 때 객체는 참조 카운트가 0이되고 즉시 메모리에서 사라짐)
            # 그러나 코드는 오류없이 동작하긴 하지만, 순환참조 중인 변수에 None을 대입해도 메모리상에서 지워지지 않음. (참조 카운트가 0이 되지 않았기 때문)
            # 이걸 해결하기 위해 output을 weakref로 참조함.weakref.ref(output)을 하면 output 이 self.outputs 리스트 안에 들어가긴 하는데, output의 ref count를 올리지 않음.

            ## weakref 를 사용하면,
            # 대신, weakref로 정의된 outputs의 element들을 사용할 때는 뒤에 ()처럼 호출해야 함. ex. for output in function.outputs: print(output())
            # 파이썬 내부적으로 ref count=0이 되면 메모리에서 변수 삭제. 그 다음 self.outputs를 부르면 [None, None ...] 나올 것.

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

    with no_grad():
        x = Variable(np.array(2.0))
        y = square(x)

    x = Variable(np.array([[1,2,3], [4,5,6]]))
    print(len(x))

    print(x)
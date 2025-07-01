import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    # gradient 저장 (backprop 때 사용)
        self.creator = None # 함수는 변수의 creator (backprop 자동화 때 사용)
    
    def set_creator(self, func):
        self.creator = func

# 기반 클래스
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output= Variable(y)
        output.set_creator(self)    # 출력 변수에 창조자 설정
        self.input = input # 입력 변수를 기억 (backprop 때 사용)
        self.output = output # 출력 저장 (왜?)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy): # gy는 출력 쪽에서 전해지는 미분값
        x = self.input.data
        gx = 2 * x * gy # Square 함수를 해석적으로 미분한 결과
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# centered difference 구현. 오차 발생 가능성 O (주로 자릿수 누락 때문)
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)


if __name__ == '__main__':

    x = Variable(np.array(0.5))

    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    y = f(x)
    print(type(y))
    print(y.data)

    dy = numerical_diff(f, x)
    print(dy)
    
    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    x.grad = A.backward(B.backward(C.backward(y.grad)))
    print(x.grad)

    assert y.creator == C
    assert y.creator.input == b
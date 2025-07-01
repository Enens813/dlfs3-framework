import numpy as np

class Variable:
    def __init__(self, data):
        # np.array만 받도록 강제
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은(는) 지원하지 않습니다".format(type(data)))
            
        self.data = data
        self.grad = None    # gradient 저장 (backprop 때 사용)
        self.creator = None # 함수는 변수의 creator (backprop 자동화 때 사용). creator는 계산할 때 정해짐(Define-by-run)
    
    def set_creator(self, func):
        self.creator = func

    # backprop 자동화를 위해 변수에 backprop 함수 적용. 그 앞의 함수를 backprop 시키고, 그 앞의 함수의 input을 backprop 시키고 ...
    def backward(self):
        # grad가 없으면 1을 넣어라 (최종 output의 초기값 설정)
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)  # 함수의 backward 호출

            if x.creator is not None:
                funcs.append(x.creator)

        # 재귀를 이용한 방법: 메모리 등 이유로 불안정
        # f = self.creator
        # if f is not None:
        #     x = f.input
        #     x.grad = f.backward(self.grad)
        #     x.backward() # 하나 앞 변수의 backward 를 호출 (재귀)

def as_array(x): # ndarray에 scalar 연산을 하면 scalar를 내보내는 numpy 특성상, scalar는 다시 array로 바꿔주어야 함.
    if np.isscalar(x):
        return np.array(x)
    return x

# 기반 클래스
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output= Variable(as_array(y)) # numpy만 이용할 경우, as_array함수 필요 (함수설명 주석 참조)
        output.set_creator(self)    # 출력 변수에 창조자 설정
        self.input = input # 입력 변수를 기억 (backprop 때 사용)
        self.output = output # 출력 저장 (backprop 재귀를 stack으로 바꿀 때 사용)
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

    ## 1. numerical difference 계산
    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    y = f(x)
    dy = numerical_diff(f, x)
    print(type(y))
    print(y.data)
    print(dy)
    
    ## 2. backprop으로 계산
    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    y.backward()

    # numerical difference와 결과가 같게 나옴.
    print(x.grad)

    # creator, input 관계 확인
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    ## 3. 함수 class의 진짜 함수화
    def square(x):
        f = Square()
        return f(x)
    def exp(x):
        f = Exp()
        return f(x)

    a = square(x)
    b = exp(a)
    y = square(b)
    y.backward()
    print(x.grad)


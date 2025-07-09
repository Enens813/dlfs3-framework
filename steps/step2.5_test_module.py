import sys, os
try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base = os.getcwd()
sys.path.append(base)

## python이 모듈을 임포트 할 때 sys.path에서 찾음. 여기에서 아래의 순서대로 찾는데,
# 1. 현재 실행 중인 파일이 있는 디렉토리
# 2. PYTHONPATH 환경 변수에 설정된 경로
# 3. 표준 라이브러리 경로
# 4. site-packages (pip 설치된 패키지들)
# 1은 파일위치 디렉토리이므로, dezero폴더가 steps 안에 있으면 찾을 수 있음. 그러나 dezero는 그밖에 있음 -> 저 리스트 안에 없음 -> sys.path에  경로를 추가해줘야 함
# 이 변경은 일시적이고, 영구적으로 바꾸려면 terminal에서 PYTHONPATH에 추가해줘야 함
# try~except는 .py파일에선 항상 try가 실행되지만, jupyter등에선 오류남. 그 때 부모 폴더의 위치를 가져오도록 설정

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x+3) ** 2
y.backward()
print(y, x.grad)

# Test functions for optimization
def sphere(x,y):
    z = x**2 + y**2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z1 = sphere(x,y)
z1.backward()
print(x.grad, y.grad)

x.cleargrad()
y.cleargrad()
z2 = matyas(x,y)
z2.backward()
print(x.grad, y.grad)

x.cleargrad()
y.cleargrad()
z3 = goldstein(x,y)
z3.backward()
print(x.grad, y.grad)


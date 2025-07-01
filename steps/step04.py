from steps.step01-03 import Variable

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data+eps)
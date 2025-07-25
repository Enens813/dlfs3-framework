import os
import subprocess
import numpy as np
# from core_simple import as_variable

# =============================================================================
# Visualize for computational graph
# =============================================================================

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)

def _dot_func(f):
    # for function
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret

# backward 호출순서와 거의비슷(generation은 여기선 무관해서) 하게 하나씩 graph text에 추가함
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    
    return f'digraph g {{\n{txt} }}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.Desktop.dlfs3-framework.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:]  # Extension(e.g. png, pdf)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass



# =============================================================================
# dezero.functions를 도와주는 함수
# =============================================================================

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim            # lead만큼 dimension을 줄여야 함
    lead_axis = tuple(range(lead))  # sum 해야 하는 axis들

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])   # shape에서 1인 축은 broadcasting되었을 축 → sum으로 되돌려야 함
    y = x.sum(lead_axis + axis, keepdims=True)                          # sum 수행
    if lead > 0:
        y = y.squeeze(lead_axis)                                        # 앞쪽 여분 차원을 제거 (squeeze)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)

    # axis를 tuple로 통일
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):  # sum() 결과가 차원을 없앤 경우
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)                              # shape 추가
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


if __name__ == '__main__':
    from core_simple import *
    
    setup_variable()

    x = Variable(np.random.randn(2,3))
    x.name = 'x'
    print(_dot_var(x))
    print(_dot_var(x, verbose=True))

    x1 = Variable(np.array(1.0))
    y = x + x1
    txt = _dot_func(y.creator)
    print(txt)
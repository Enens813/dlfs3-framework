# 이 파일이 있어야 python이 모듈로 인식. 내용 없으면 빈 파일로라도 만들어야 함.
# 또한, 모듈을 임포트할 때 첫 번째로 호출되는 코드

is_simple_core = False # True

if is_simple_core:
    from dezero.core_simple import Variable, Function, using_config, no_grad, as_array, as_variable # 이렇게 해야 from dezero import Variable 할 수 있음. (원랜 from dezero.core_simple import Variable)
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable, Function, using_config, no_grad, as_array, as_variable
    from dezero.core import setup_variable

import dezero.functions

setup_variable()
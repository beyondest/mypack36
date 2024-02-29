from autoaim_alpha.autoaim_alpha.os_op.decorator import *

@timing(1)
def newton_method(f, df, x0, tol=1e-6, max_iter=100):
    """
    使用牛顿法求解方程 f(x) = 0 的根。

    参数：
    f: 目标函数
    df: 目标函数的导数
    x0: 初始猜测值
    tol: 允许的误差
    max_iter: 最大迭代次数

    返回值：
    x: 方程的根
    """
    x = x0
    for i in range(max_iter):
        x_new = x - f(x) / df(x)
        print(f"第 {i+1} 次迭代，abs(x - x_new) = {abs(x - x_new)}")
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("牛顿法未收敛")

# 示例：求解方程 x^2 - 2 = 0
def f(x):
    return x**2 - 2

def df(x):
    return 2*x

root, t = newton_method(f, df, 1.5)
print("方程的根为：", root)
print("运行时间：", f'{t:.5f}')

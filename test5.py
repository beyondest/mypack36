import numpy as np
import matplotlib.pyplot as plt

# 定义常量
g = 9.8  # 重力加速度，单位 m/s^2
k = 0.1  # 空气阻力系数，单位 Ns/m
m = 1.0  # 小球质量，单位 kg
v0 = np.array([1.0, 1.0])  # 初始速度，单位 m/s
dt = 0.01  # 时间步长，单位 s
t_max = 10.0  # 最大模拟时间，单位 s

# 初始化变量
t = 0.0
x = np.array([0.0, 0.0])  # 初始位置，单位 m
v = v0  # 初始速度，单位 m/s

# 存储结果
trajectory = []

# 欧拉法求解
while t < t_max:
    # 计算空气阻力
    f = -k * v
    
    # 计算加速度
    a = np.array([0.0, -g]) + f / m
    
    # 更新速度和位置
    v = v + a * dt
    x = x + v * dt
    
    # 存储结果
    trajectory.append(x.copy())
    
    # 更新时间
    t = t + dt

# 将结果转换为 NumPy 数组
trajectory = np.array(trajectory)

# 绘制运动轨迹
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectory of the ball')
plt.grid(True)
plt.show()

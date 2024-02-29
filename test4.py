import math
import numpy as np
# 定义旋转向量
y_unit  = np.array([0, 1, 0])
rot_vec = np.array([0, -math.pi/2, 0])
rot_vec = rot_vec - y_unit * np.pi/2

v = np.array([1, 0, 0]).reshape(3, 1)
# 将旋转向量转换为旋转矩阵


# 将旋转向量转换为旋转矩阵

def TRANS_RVEC_TO_ROT_MATRIX(rvec:np.ndarray)->np.ndarray:

    theta = np.linalg.norm(rot_vec)
    k = rot_vec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return rot_matrix



np.set_printoptions(precision=3, suppress=True)

rot_mat = TRANS_RVEC_TO_ROT_MATRIX(rot_vec)
print(rot_mat @ v)


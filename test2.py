import cv2
import numpy as np
from autoaim_alpha.os_op.basic import TRANS_RVEC_TO_ROT_MATRIX
# 3D点坐标
object_points = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 1, 0]], dtype=np.float32)

# 2D点坐标
image_points = np.array([[10, 10],
                         [20, 10],
                         [10, 20],
                         [20, 20]], dtype=np.float32)

object_points = TRANS_RVEC_TO_ROT_MATRIX(np.array([1, np.pi/2, 1])) @ object_points.T
object_points = object_points.T


# 相机内参矩阵
mtx = np.array([[100, 0, 50],
                [0, 100, 50],
                [0, 0, 1]], dtype=np.float32)

# 相机畸变矩阵
dist = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# 使用solvePnP函数估计相机的姿态
retval, rvec, tvec = cv2.solvePnP(object_points, image_points, mtx, dist)
np.set_printoptions(suppress=True)
# 打印相机的姿态
print("Rotation vector:")
print(rvec)
print("Translation vector:")
print(tvec)

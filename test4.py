import cv2
import numpy as np

# 创建一个空白图像
image = np.zeros((512, 512, 3), np.uint8)

# 列表a包含10个nparray，shape为（2，）
a = [np.array([10, 20]), np.array([30, 40]), np.array([50, 60]), np.array([70, 80]), np.array([90, 100]), 
     np.array([110, 120]), np.array([130, 140]), np.array([150, 160]), np.array([170, 180]), np.array([190, 200])]

# 将数组转换为图像
points = np.array(a, np.int32)
points = points.reshape((-1, 1, 2))

# 连接点成曲线
cv2.polylines(image, [points], False, (0, 255, 0), 2)

# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np



object_points = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 1, 0]], dtype=np.float32)

image_points = np.array([[10, 20],
                         [50, 20],
                         [10, 80],
                         [50, 80]], dtype=np.float32)


camera_matrix = np.array([[1000, 0, 320],
                         [0, 1000, 240],
                         [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))

retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

print(retval)
print(np.round(rvec))
print(np.round(tvec))

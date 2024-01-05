import cv2
import numpy as np
import sys
sys.path.append('..')
import img.img_operation as imo
img = cv2.imread(imo.RED_PATH)
def on_trackbar(value):
    global img
    lower = cv2.getTrackbarPos('Lower Value', 'config')
    dst=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    y,u,v=cv2.split(dst)
    dst=cv2.inRange(v,lower,255)
    
    cv2.imshow('config', dst)

cv2.namedWindow('config',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Lower Value', 'config', 0, 255, on_trackbar)

cv2.imshow('config', img)

while True:
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

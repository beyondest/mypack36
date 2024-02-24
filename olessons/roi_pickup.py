
import sys
sys.path.append('../img/img_operation')
sys.path.append('..')


import img.img_operation as imo
import cv2
import numpy as np
path='./res/armorred.png'
img=cv2.imread(path)
imo.cvshow(img)
dst,_=imo.pre_process(img,'red')
imo.cvshow(dst)
rec_list,_=imo.find_big_rec(dst)
center_list = imo.turn_big_rec_list_to_center_points_list(rec_list)
print(center_list)
cv2.circle(img,center_list[0],10,(255,0,0),-1)
imo.cvshow(img)
roi_list,_=imo.pick_up_roi(rec_list,img)
roi_single_list,_=imo.pre_process2(roi_list,'red')
cv2.imshow('roi',roi_single_list[0])
cv2.waitKey(0)
cv2.destroyAllWindows()





roi_transform_list,_=imo.pick_up_roi_transform(rec_list,img)

roi_single_list,_=imo.pre_process3(roi_transform_list,'red')

cv2.imshow('d',roi_single_list[0])
cv2.waitKey(0)
cv2.destroyAllWindows()




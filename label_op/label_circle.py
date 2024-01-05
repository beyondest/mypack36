import cv2
import os
import numpy as np
import os_operation as oso
video_path='D:/tmdvs/pywork/RM/hei'
frame_path='D:/tmdvs/pywork/RM/armor/baiche/frame'
bin_path='D:/tmdvs/pywork/RM/armor/baiche/bin'
trans_path='D:/tmdvs/pywork/RM/armor/baiche/trans'
def hand_trans(root_path:str,out_path:str):
    
    path_list=os.listdir(root_path)
    for i in path_list:
        abs_path=os.path.join(root_path,i)
        ori_name=oso.get_name(abs_path)
        img = cv2.imread(abs_path)  
        p_list = []  #leftup,rightup,leftdown,rightdown
        img2 = img.copy()
        cv2.namedWindow("original_img", cv2.WINDOW_NORMAL)
        cv2.imshow("original_img", img)
        cv2.setMouseCallback('original_img',capture_event,[img,img2,p_list,out_path,ori_name])
        cv2.waitKey(0) 
        continue
        
def capture_event(event, x, y, flags,param_list:list):
    img, img2,p_list,out_path,ori_name=param_list
    if event == cv2.EVENT_LBUTTONDOWN:
        # create a circle at that position
        # of radius 30 and color greeen
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("original_img", img)
        p_list.append([x, y])
        if len(p_list) == 4:
            # target size
            dst_point = (1000, 1000) 
            pts1 = np.float32(p_list)
            pts2 = np.float32([[0, 0], [dst_point[0], 0], [0, dst_point[1]], [dst_point[0], dst_point[1]]])
            dst = cv2.warpPerspective(img2, cv2.getPerspectiveTransform(pts1, pts2), dst_point)
            cv2.imwrite(os.path.join(out_path,ori_name+'trans.png'), dst)
            cv2.destroyWindow('original_img')
hand_trans(frame_path,trans_path)
            





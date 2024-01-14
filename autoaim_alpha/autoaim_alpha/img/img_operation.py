import cv2
import os
import numpy as np
from threading import Thread
from time import sleep,ctime
import random
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('..')
from ..os_op import os_operation as oso
from ..os_op.decorator import *
from typing import Optional,Union
#yuv_range:
#red:152-255,v
#blue:145-255,u

PRESS_ESC=(cv2.waitKey(1) & 0xFF ==27)
RED_PATH='/home/liyuxuan/vscode/res/armor/armorred.png'
def target_find(img_ori:np.ndarray,armor_color:str,filter_params:list)->tuple:
    '''
    Function:\n
    NO1:find light_bar\n
    NO2:make big rec to cover two light_bar\n
    NO3:use big rec to make a mask to select roi\n
    notice that there may be a lot of roi,so return list\n
    NO4:calculate time and show FPS\n
    return draw_img,roi_single_list,all_time
    
    '''
    img_single,t1=pre_process(img_ori,armor_color)
    big_rec_list,t2=find_big_rec_plus(img_single,filter_params)
    draw_img=draw_big_rec(big_rec_list,img_ori)
    roi_list,t3=pick_up_roi(big_rec_list,img_ori)
    roi_single_list,t4=pre_process2(roi_list,armor_color)
    all_time=t1+t2+t3+t4
    return draw_img,roi_single_list,all_time

def find_armor( img_bgr:np.ndarray,
                img_bgr_exposure2:np.ndarray,
                color = 'red',
                if_debug:bool = True
                )->list:
    """Get center list and roi transform list

    Args:
        img_bgr (np.ndarray): _description_
        img_bgr_exposure2 (np.ndarray): _description_
        color (str, optional): _description_. Defaults to 'red'.

    Returns:
        [center_list, roi_transfomr_list, all_spend_time]
    """
    img_single,pre_process_time = pre_process(          img_bgr=img_bgr,
                                                        armor_color=color)
    if if_debug:
        cv2.imshow('single_after_preprocess',img_single)
    big_rec_list ,find_big_rec_time= find_big_rec(img_single,trace=True)
    if if_debug:
        draw_big_rec(big_rec_list,img_bgr,True)
    center_list = turn_big_rec_list_to_center_points_list(big_rec_list)
    if if_debug:
        draw_center_list(center_list,img_bgr)
        
    roi_transform_list,pick_up_roi_transform_time = pick_up_roi_transform(big_rec_list,img_bgr_exposure2)
    if if_debug and roi_transform_list is not None and len(roi_transform_list) == 1:
        cv2.imshow('roi_transform',roi_transform_list[0])
    
    roi_single_list , pre_process2_time = pre_process2(roi_transform_list,'red')
    if if_debug and roi_single_list is not None and len(roi_single_list) == 1:
        cv2.imshow('roi_single',roi_single_list[0])
    
    
    if if_debug:
        print('pre_process_time',pre_process_time)
        print('find_big_rec_time',find_big_rec_time)
        print('pickup_roi_transfomr_time',pick_up_roi_transform_time)
    
    return center_list,roi_single_list,pre_process_time+find_big_rec_time+pick_up_roi_transform_time+pre_process2_time

###############################################################

@timing(1)
def pre_process(img_yuv:np.ndarray,armor_color:str)->Union[list,None]:
    '''
    @timing
    armor_color = 'red' or 'blue'\n
    return img_single,time \n
    Warning:  input img is YUV !!!
    '''
    if img_yuv is None:
        return None
    img_size_yx=(img_yuv.shape[0],img_yuv.shape[1])
    

    dst=cv2.GaussianBlur(img_yuv,(3,3),1)
    
    #out of memory error
    y,u,v=cv2.split(dst)
    #blue 145-255
    #red 152-255
    if armor_color=='blue':
        dst=cv2.inRange(u.reshape(img_size_yx[0],img_size_yx[1],1),145,255)
    elif armor_color=='red':
        dst=cv2.inRange(v.reshape(img_size_yx[0],img_size_yx[1],1),152,255)
    else:
        print('armor_color is wrong')
        sys.exit()
        
    #dst=cv2.medianBlur(dst,13)
    
    if img_size_yx==(1024,1280):
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    elif img_size_yx==(256,320):
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    else:
        print('imgsize not match, you have to preset preprocess params ')  
        sys.exit() 
    dst=cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel)
    return dst






def find_big_rec_plus(img_single:np.ndarray,
                      filter_params:tuple
                      )->tuple:
    '''
    add trackbar_set\n
    filter_params:
    @params[0]: area_range
    @params[1]: normal_ratio
    @params[2]: strange_ratio1
    @params[3]: strange_ratio2
    @params[4]: shape_like_range
    @params[5]: center_dis_range
    '''
    img_size_yx=(img_single.shape[0],img_single.shape[1])
    out_list=[]
    final_list=[]
    
    
    
    time1=cv2.getTickCount()
    conts,arrs=cv2.findContours(img_single,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
            
    conts=filter_area(conts,filter_params[0])
    #print(len(conts))
    conts=filter_normal(conts,ratio=filter_params[1])
    #print(len(conts))
    conts=filter_strange(conts,ratio=filter_params[2])
    #print(len(conts))

    conts_tuple_list=filter_no_shapelike(conts,filter_params[4],filter_params[4])
    #print(len(conts_tuple_list))
    for i in conts_tuple_list:
        
        if is_center_dis_good(i[0],i[1],filter_params[5]):
        
            big_rec_info=make_big_rec(i[0],i[1])
            #wid and hei cant < 1
            if big_rec_info[2]>1 and big_rec_info[3]>1:
                
                out_list.append(big_rec_info[4])
        
        
        
    out_list=filter_strange(out_list,filter_params[3])
    out_list=expand_rec_wid(out_list,expand_rate=2,img_size_yx=img_size_yx) 
    
   
        
        
    time2=cv2.getTickCount()
    time=(time2-time1)/cv2.getTickFrequency()
    return out_list ,time 

@timing(1)
def find_big_rec(img_single:np.ndarray,trace:bool=True)->list:
    '''
    @timing\n
    big_rec is expanded already\n
    return big_rec_list,time
    '''
    if img_single is None:
        return None
    img_size_yx=(img_single.shape[0],img_single.shape[1])
    out_list=[]
    if img_size_yx==(1024,1280):
        mode='big'
    elif img_size_yx==(256,320):
        mode='small'
    else:
        print('img_size not match, you havent preset params of find_big_rec yet')
        sys.exit()
    
    conts,arrs=cv2.findContours(img_single,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if trace:
        print('findcontours:',len(conts))
    if mode=='big':
        
        if len(conts)!=-1:
            
            conts=filter_area(conts,(500,2500))
            if trace:
                print('after filter_area:',len(conts))
            conts=filter_normal(conts,ratio=2)
            if trace:
                print('after filter_normal:',len(conts))
            conts=filter_strange(conts,ratio=5)
            if trace:
                print('after filter_strange:',len(conts))
            
            conts_tuple_list=filter_no_shapelike(conts,(0.1,10),(0.1,10))
            if trace:
                print('after filter_no_shapelike:',len(conts_tuple_list))
            for i in conts_tuple_list:
                
                if iscenternear(i[0],i[1],600):
                    
                    big_rec_info=make_big_rec(i[0],i[1])
                    #wid and hei cant < 1
                    if big_rec_info[2]>1 and big_rec_info[3]>1:
                      
                        out_list.append(big_rec_info[4])
            if trace:
                print('final bigrec(after centernear):',len(out_list))
        else:
            big_rec_info=make_big_rec(conts[0],conts[1])
            out_list.append(big_rec_info[4])

        out_list=expand_rec_wid(out_list,expand_rate=2,img_size_yx=img_size_yx) 
    
    elif mode=='small':
        #small_params = 1/4 * big_params
        if len(conts)!=-1:
            
            conts=filter_area(conts,(1,400))
            if trace:
                print('after filter_area:',len(conts))
            conts=filter_normal(conts,ratio=1.2)
            if trace:
                print('after filter_normal:',len(conts))
            conts=filter_strange(conts,ratio=10)
            if trace:
                print('after filter_strange:',len(conts))
            conts_tuple_list=filter_no_shapelike(conts,(0.2,1.8),(0.2,1.8))
            if trace:
                print('after filter_no_shapelike:',len(conts_tuple_list))
            for i in conts_tuple_list:
                
                if iscenternear(i[0],i[1],80):
                    
                    big_rec_info=make_big_rec(i[0],i[1])
                    
                    #wid and hei cant <1
                    if big_rec_info[2]>1 and big_rec_info[3]>1:
                    
                        out_list.append(big_rec_info[4])
            if trace:
                print('final bigrec(after centernear):',len(out_list))
                
        else:
            big_rec_info=make_big_rec(conts[0],conts[1])
            out_list.append(big_rec_info[4])
        
    
        out_list=expand_rec_wid(out_list,expand_rate=2,img_size_yx=img_size_yx)  
        
        

    return out_list  

def draw_big_rec(big_rec_list,img_bgr:np.ndarray,if_draw_on_input:bool = True)->np.ndarray:
    '''
    return img_copy_bgr
    '''
    if big_rec_list is None:
        return None
    if img_bgr.shape[2]==1:
        img_bgr=cv2.cvtColor(img_bgr,cv2.COLOR_GRAY2BGR)
    if if_draw_on_input:
        img_copy = img_bgr
    else:
        img_copy=img_bgr.copy()
    for i in big_rec_list:
        draw_cont(img_copy,i,2,3)
    return img_copy
    
def draw_center_list(center_list:list,img_bgr:np.ndarray, if_draw_on_input:bool = True) ->np.ndarray:
    if center_list is None:
        return None
    radius = round((img_bgr.shape[0]+img_bgr.shape[1])/200)
    color = (0,255,0)
    if if_draw_on_input:
        
        for i in center_list:
            cv2.circle(img_bgr,i,radius,color,-1)
            
        return img_bgr
    
    else:
        img = img_bgr.copy()
        for i in center_list:
            cv2.circle(img,center_list,radius,color,-1)
            
        return img
  
def pick_up_roi(big_rec_list,img_ori:np.ndarray)->list:
    '''
    return roi_list,time\n
    if nothing find, roi_list[0] is black canvas of img_ori.shape
    '''
    img_size_yx=(img_ori.shape[0],img_ori.shape[1])
    t1=cv2.getTickCount()
    roi_list=[]
    
    background=np.zeros((img_size_yx[0],img_size_yx[1]),dtype=np.uint8)
    for i in big_rec_list:
        i=i.reshape(-1,1,2)
        back_copy=background.copy()
        img_copy=img_ori.copy()
        dst=img_copy
        mask=cv2.fillPoly(back_copy,[i],255)
        dst=cv2.bitwise_and(img_copy,img_copy,mask=mask)
        roi_list.append(dst)
    if len(big_rec_list)==0:
        bgr=cv2.cvtColor(background,cv2.COLOR_GRAY2BGR)
        roi_list.append(bgr)
    t2=cv2.getTickCount()
    time= (t2-t1)/cv2.getTickFrequency()
    return roi_list,time


@timing(1)
def pick_up_roi_transform(rec_list:list,img_ori:np.ndarray)->list:
    '''
    @timing \n
    return roi_transform_list, notice they are bgr images\n 
    '''
    if rec_list == None:
        return None
    
    roi_transform_list=[]
    
    for i in rec_list:
        
        _,_,wid,hei,_,_=getrec_info(i)
        i=Img.Change.order_rec_points(i)
        dst_points=np.array([[0,0],[wid-1,0],[wid-1,hei-1],[0,hei-1]],dtype=np.float32)
        M=cv2.getPerspectiveTransform(i.astype(np.float32),dst_points)
        dst=cv2.warpPerspective(img_ori,M,(int(wid),int(hei)),flags=cv2.INTER_LINEAR)
        roi_transform_list.append(dst)
    return roi_transform_list
        

@timing(1)
def pre_process2(roi_transform_list:list,armor_color:str='red',strech_max:Optional[int]=None)->list:
    '''
    @timing\n
    preprocess for distinguish number 
    return roi_single_list,time
    '''
    if roi_transform_list is None:
        return None
    roi_single_list=[]
    for i in roi_transform_list:
        dst=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)

        dst=gray_stretch(dst,strech_max)

        if armor_color=='red':
            ret,dst=cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
        elif armor_color=='blue':    
            ret,dst=cv2.threshold(dst,90,255,cv2.THRESH_BINARY)
            
        roi_single_list.append(dst)
        
        
    return roi_single_list
    
    
    

#####################################################################


def cvshow(img:np.ndarray,windows_name:str='show'):
    '''
    use to show quickly
    '''
    cv2.imshow(windows_name,img)
    while True:
        if (cv2.waitKey(0) & 0xff )== 27:
            break
    cv2.destroyAllWindows()
    
def cvshow2(abs_path:str,nothing:None):
    
    img = cv2.imread(abs_path)
    cv2.imshow('d',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

def plt_show0(img):
    '''
    show 3 channels bgr in plt
    '''
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()
    
def plt_show(img):
    '''
    show single channel img
    '''
    plt.imshow(img,cmap='gray')
    plt.show()

def add_text(img_bgr:np.ndarray,
             name:str,
             value,
             pos:tuple=(-1,-1),
             font:int=cv2.FONT_HERSHEY_SIMPLEX,
             color:tuple=(0,0,0),
             scale_size: float= 0.5
             )->np.ndarray:
    
    '''
    show name:value on the position of pos(x,y)\n
    if pos =(-1,-1), then auto position\n
    return img_bgr
    '''
    img_size_yx=(img_bgr.shape[0],img_bgr.shape[1])
    if pos==(-1,-1):
        
        pos=(round(img_size_yx[1]/10),round(img_size_yx[0]/10))
    
    
    thickness=round(3/(1024*1280)*(img_size_yx[0]*img_size_yx[1]))
    dst=cv2.putText(img_bgr,f'{name}:{value}',pos,font,scale_size,color,thickness)
    return dst
    
###################################################################
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def get_edgebin2(ori_img:np.ndarray)->np.ndarray:
    '''by canny,50-100'''
    img_copy=ori_img.copy()
    img_gray=gray_guss(img_copy)
    img_blur=cv2.GaussianBlur(img_gray,(5,5),1)
    img_canny=cv2.Canny(img_blur,50,100)
    img_canny=cv2.GaussianBlur(img_canny,(3,3),1)
    ret,img_edge_bin=cv2.threshold(img_canny,0,255,cv2.THRESH_OTSU)
    img_edge_bin=cv2.medianBlur(img_edge_bin,3)
    return img_edge_bin

def get_edgebin1(ori_img:np.ndarray)->np.ndarray:
    '''by sobel'''
    img_copy=ori_img.copy()
    img_gray=gray_guss(img_copy)
    sobel_x=cv2.Sobel(img_gray,cv2.CV_16S,1,0,ksize=3)
    sobel_y=cv2.Sobel(img_gray,cv2.CV_16S,0,1,ksize=3)
    sobel_x=cv2.convertScaleAbs(sobel_x)
    sobel_y=cv2.convertScaleAbs(sobel_y)
    img_sobel=cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
    img_sobel=cv2.GaussianBlur(img_sobel,(3,3),1)
    ret,img_thresh=cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU)
    img_thresh=cv2.medianBlur(img_thresh,5)
    ker_rec=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    img_close=cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,ker_rec,iterations=1)

    img_edge_bin=cv2.medianBlur(img_close,5)
    return img_edge_bin
################################################################3
def make_edge(abs_path:str,out_path:str,fmt:str='png'):
    ori_img=cv2.imread(abs_path)
    edge_bin=get_edgebin1(ori_img)
    ori_name=oso.get_name(abs_path)
    cv2.imwrite(os.path.join(out_path,ori_name+'edgebin'+'.'+fmt),edge_bin)






def draw_cont(ori_copy:np.ndarray,cont:np.ndarray,color:int=2,thick:int=2):
    '''
    @cont:  np.array(left_down,left_up,right_up,right_down)
    @color=0,1,2=blue,green,red
    @thick:   defalt is 2
    '''
    colorlist=[(255,0,0),(0,255,0),(0,0,255)]
    cv2.drawContours(ori_copy,[cont],-1,colorlist[color],thick)
    return 0

def make_cont(img,thr_area:int=500):

    edge_bin=get_edgebin1(img)
    
    conts,arrs=cv2.findContours(edge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    img_copy=img.copy()
    for i in conts:
        area=cv2.contourArea(i)
        if area>thr_area:
            rec_points=getrec_info(i)[4]
            draw_cont(img_copy,rec_points,2)
    return img_copy


def getframe_info(vd_abspath:str)->tuple:
    '''fps,frame_count,frame_width,frame_height=getframe_info()'''
    vd=cv2.VideoCapture(vd_abspath)
    Fps=vd.get(cv2.CAP_PROP_FPS)
    frame_count=vd.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height=vd.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width=vd.get(cv2.CAP_PROP_FRAME_WIDTH)
    vd.release()
    return Fps,frame_count,frame_width,frame_height

def readframe(abs_path:str,
              out_path:str,
              fmt:str,
              interval:int=5,
              name:str='frame',
              start:int=0,
              length:int=0):
    '''cv2 get started from 0 frame'''
    vd_name=oso.get_name(abs_path)
    vd=cv2.VideoCapture(abs_path)
    vd.set(cv2.CAP_PROP_POS_FRAMES,start)
    if length==0:
        length=vd.get(cv2.CAP_PROP_FRAME_COUNT)
    count=start-1
    #time check
    print(f' {start} start as {ctime()}')
    while True:
        count+=1
        iscaptured=vd.grab()
        if not iscaptured or count==start+length:
            break
        if count%interval==0:
            iscaptured,frame=vd.retrieve()
            if frame is not None:
                cv2.imwrite(os.path.join(out_path,vd_name+name+'{}.'.format(int(count))+fmt),frame)  
            else:
                break
    #time check
    print(f' {start} end as {ctime()}')
    vd.release()

def readframe_pro(vd_abspath:str,
                  out_path:str,
                  fmt:str='png',
                  interval:int=10,
                  name:str='frame',
                  threads:int=6,
                  start_frame:int=100,
                  end_frame:int=4000)->None:
    '''read frame faster by multi-thread,
        frame range is (0,frame_count-1)'''
    vd=cv2.VideoCapture(vd_abspath)
    frame_count=vd.get(cv2.CAP_PROP_FRAME_COUNT)
    if end_frame == 0 or end_frame>=frame_count:
        end_frame=frame_count
    #length means how long each thread deal with
    length=(end_frame-start_frame)//threads+1
    start_list=[start_frame+i*length for i in range(threads)]
    #multi-thread, after test, 6 runs fastest,10 seconds for 200 frames,1 runs lowest, 35 seconds for 200 frames
    t=[]
    for i in range(threads):
        t.append(Thread(target=readframe,args=(vd_abspath,out_path,fmt,interval,name,start_list[i],length)))
    for i in range(threads):
        t[i].start()
    

    
    return None


def find_inorder(points_list:list)->list:
    for i in range(3):
        if points_list[i][0]>points_list[i+1][0]:
            points_list[i],points_list[i+1]=points_list[i+1],points_list[i]
    if points_list[0][1]<points_list[1][1]:
        points_list[0],points_list[1]=points_list[1],points_list[0]
    if points_list[2][1]<points_list[3][1]:
        points_list[2],points_list[3]=points_list[3],points_list[2]
    points_list[1],points_list[2]=points_list[2],points_list[1]
    return points_list
            
     
#1000,max3,max2 is best up to now
def make_findcont_transform(abs_path:str,out_path:str,fmt:str='png',thr_area:int=1000):
    p_list = []  
    dst_point = (1000, 1000)  
    img=cv2.imread(abs_path)
    img_size_yx=(img.shape[0],img.shape[1])
    img2=img.copy()
    ori_name=oso.get_name(abs_path)
    edge_bin=get_edgebin1(img)
    rec_points=[]
    conts,arrs=cv2.findContours(edge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for i in conts:
        area=cv2.contourArea(i)
        if area>thr_area:
            rec_points.append(getrec_info(i)[4])
    if len(rec_points)==0:
        wid=img_size_yx[1]
        hei=img_size_yx[0]
        x1=random.randint(0,wid//3)
        y1=random.randint(0,hei//3)
        x2=random.randint(wid-wid//3,wid-1)
        y2=random.randint(0,hei//3)
        x3=random.randint(0,wid//3)
        y3=random.randint(hei-hei//3,hei-1)
        x4=random.randint(wid-wid//3,wid-1)
        y4=random.randint(hei-hei//3,hei-1)
        p_list =[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    elif len(rec_points)==1:
        
        p_list= [rec_points[0][0],rec_points[0][1],rec_points[0][3],rec_points[0][2]]
        p_list=find_inorder(p_list)
    elif len(rec_points)>1:
        area_list=[]
        for i in rec_points:
            area_list.append(cv2.contourArea(i))
        compare_list=sorted(area_list,reverse=True)
        max1=area_list.index(compare_list[0])
        max2=area_list.index(compare_list[1])
        max1=rec_points[max1]
        max2=rec_points[max2]
        
        if len(rec_points)>2:
            max3=area_list.index(compare_list[2])
            max3=rec_points[max3]
            fine=max3
            p_list= [fine[0],fine[1],fine[3],fine[2]]
            p_list=find_inorder(p_list)
        else:
            fine=max2
            p_list= [fine[0],fine[1],fine[3],fine[2]]
            p_list=find_inorder(p_list)
    pts1 = np.float32(p_list)
    pts2 = np.float32([[0, 0], [dst_point[0], 0], [0, dst_point[1]], [dst_point[0], dst_point[1]]])
    dst = cv2.warpPerspective(img2, cv2.getPerspectiveTransform(pts1, pts2), dst_point)
    dst=cv2.flip(dst,0)
    cv2.imwrite(os.path.join(out_path,ori_name+'trans'+'.'+fmt),dst)

def make_bin(abs_path:str,out_path:str,fmt:str='png',name:str='bin'):
    img = cv2.imread(abs_path)
    ori_name=oso.get_name(abs_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, img_bin= cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(out_path,ori_name+name+'.'+fmt),img_bin)

def add_noise_circle(img:np.ndarray,
               circle_radius_ratio:float = 1/7,
               noise_probability:float = 1/3
               )->np.ndarray:
    '''Make random white circle as noise, the radius of it is 
    1/7 of height, 1/3 possibility add noise'''
    judge=random.randint(1,round(1/noise_probability))
    if judge==1:
        height=img.shape[0]
        wid=img.shape[1]
        radius=height//round(1/circle_radius_ratio)
        x=random.randint(0,wid-1)
        y=random.randint(0,height-1)
        cv2.circle(img,(x,y),radius,(255,255,255),-1)  
    return img


def add_noise_strip_toedge(img:np.ndarray,
                           strip_width_ratio:float = 1/7,
                           strip_length_ratio:float = 1,
                           noise_probablity = 1,
                           ):  
    judge = random.randint(1,round(1/noise_probablity))
    if judge == 1:
        rows, cols, _ = img.shape
        noise = np.zeros_like(img)
        offset_x = round((cols-strip_length_ratio * cols)/2)
        offset_y = round((rows-strip_length_ratio * rows)/2)
        strip_width_vertical = round(strip_width_ratio * cols/2)
        strip_width_horizontal = round(strip_width_ratio * rows)
        
        # add left vertical strip
        noise[0+offset_y:rows-offset_y, :strip_width_vertical, :] = 127
        
        # add right vertical strip
        noise[0+offset_y:rows-offset_y, -strip_width_vertical:, :] = 127
        
        # add up horizontal strip
        noise[:strip_width_horizontal, 0+offset_x:cols-offset_x, :] = 127
        
        # add down horizontal strip
        noise[-strip_width_horizontal:, 0+offset_x:cols-offset_x, :] = 127

        
        img = img + noise

        img = np.clip(img, 0, 255)
        
    return img




    
def trans_random_affine0(img, stretch_factor=0.5):
    
    height, width = img.shape[:2]
    

    stretch_horizontal = np.random.uniform(1 - stretch_factor, 1 + stretch_factor)

    stretch_vertical = np.random.uniform(1 - stretch_factor, 1 + stretch_factor)

    matrix = np.array([[stretch_horizontal, 0, 0], [0, stretch_vertical, 0]], dtype=np.float32)

    img_stretched = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    return img_stretched



def trans_random_affine1(img:np.ndarray, perspective_factor:float=0.5, affine_factor:float=0.5):
    """Effect not work as expected, do not use this function

    Args:
        img (np.ndarray): _description_
        perspective_factor (float, optional): _description_. Defaults to 0.5.
        affine_factor (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    height, width = img.shape[:2]

    perspective_pts = np.array(
        [
            [np.random.uniform(-perspective_factor, perspective_factor) * width, np.random.uniform(-perspective_factor, perspective_factor) * height],
            [np.random.uniform(1 - perspective_factor, 1 + perspective_factor) * width, np.random.uniform(-perspective_factor, perspective_factor) * height],
            [np.random.uniform(-perspective_factor, perspective_factor) * width, np.random.uniform(1 - perspective_factor, 1 + perspective_factor) * height],
            [np.random.uniform(1 - perspective_factor, 1 + perspective_factor) * width, np.random.uniform(1 - perspective_factor, 1 + perspective_factor) * height],
        ],
        dtype=np.float32,
    )
    

    affine_pts = np.array(
        [
            [np.random.uniform(-affine_factor, affine_factor) * width, np.random.uniform(-affine_factor, affine_factor) * height],
            [np.random.uniform(1 - affine_factor, 1 + affine_factor) * width, np.random.uniform(-affine_factor, affine_factor) * height],
            [np.random.uniform(-affine_factor, affine_factor) * width, np.random.uniform(1 - affine_factor, 1 + affine_factor) * height],
        ],
        dtype=np.float32,
    )

    matrix_perspective = cv2.getPerspectiveTransform(perspective_pts, np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32))
    matrix_affine = cv2.getAffineTransform(affine_pts, np.array([[0, 0], [width, 0], [0, height]], dtype=np.float32))

    img_perspective = cv2.warpPerspective(img, matrix_perspective, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    img_perspective_affine = cv2.warpAffine(img_perspective, matrix_affine, (width, height), borderMode=cv2.BORDER_REFLECT_101)

    return img_perspective_affine






def make_rotate(abs_path:str,
                out_path:str,
                angel:int=180,
                fmt:str='png'
                ):
    '''angel=90 clockwise/-90 counterclockwise/180'''
    if angel==90:
        cvcode=cv2.ROTATE_90_CLOCKWISE
    elif angel==-90:
        cvcode=cv2.ROTATE_90_COUNTERCLOCKWISE
    elif angel==180:
        cvcode=cv2.ROTATE_180
    img=cv2.imread(abs_path)
    img_rotate=cv2.rotate(img,cvcode)
    ori_name=oso.get_name(abs_path)
    out=os.path.join(out_path,ori_name+'.'+fmt)
    cv2.imwrite(out,img_rotate)
    
def rename_distinguish(abs_path:str,out_path:str,fmt:str='png'):
    '''shape[1]>shape[0] is big, if use muitiwork, it out_path 
    must be configed , cause even out_name='',it has '\\' in last 
    location'''
    img=cv2.imread(abs_path)
    if img.shape[1]>img.shape[0]:
        out_name='d'
        
    else:
        out_name='x'
        
    dir_path=os.path.split(out_path)[0]
    out=dir_path+out_name   
    ori_name=oso.get_name(abs_path)
    out=os.path.join(out,ori_name+out_name+'.'+fmt)
    os.rename(abs_path,out)
    
def make_cut(abs_path:str,out_path:str,fmt:str='png',suffix_name:str='cut'):
    img=cv2.imread(abs_path)
    ori_name=oso.get_name(abs_path)
    wid=img.shape[1]
    k=145/693
    cut_wid=round(wid*k)
    img2=img.copy()
    dst=img2[:,cut_wid:wid-cut_wid,:]
    out=os.path.join(out_path,ori_name+suffix_name+'.'+fmt)
    cv2.imwrite(out,dst)

##################################################################

def getrec_info(cont:np.ndarray)->tuple:
    '''
    input ori_cont,\n
    return:\n
    center_x,\n
    center_y,\n
    width,\n
    height,\n
    rec_points(its shape is same as one cont),\n
    rec_area
    '''
    ret=cv2.minAreaRect(cont)
    bo=cv2.boxPoints(ret)
    bo=np.int0(bo)
    rec_area=cv2.contourArea(bo)
    rec_area=abs(rec_area)
    return (ret[0][0],ret[0][1],ret[1][0],ret[1][1],bo,rec_area)
    
def getimg_info(abs_path:str)->tuple:
    '''return img_size,shape,dtype'''
    img=cv2.imread(abs_path)
    img_size=img.size
    img_shape=img.shape
    img_dtype=img.dtype
    return img_size,img_shape,img_dtype

def pertrans(img:np.ndarray)->np.ndarray:
    '''
    warperspective of img\n
    notice the img.shape will change
    '''
    img_size_yx=(img.shape[0],img.shape[1])
    pts1=([0,0],[img_size_yx[1],0],[0,img_size_yx[0]],[img_size_yx[1],img_size_yx[0]])
    pts1=np.float32(pts1)
    dst_point=[img_size_yx[1],img_size_yx[0]]
    pts2 = np.float32([[0, 0], [dst_point[0], 0], [0, dst_point[1]], [dst_point[0], dst_point[1]]])
    dst = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts1, pts2), dst_point)
    return dst

def image_binary(image):
    max_value=float(image.max())
    min_value=float(image.min()) 

    ret=max_value-(max_value-min_value)/2
    ret,thresh=cv2.threshold(image,ret,255,cv2.THRESH_BINARY)
    return thresh
    

def gray_stretch(image,strech_max:Optional[int]=None):
    if Img.Check.is_3channelimg(image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_float=image.astype(np.float32)
    if strech_max==None:
        
        max_value=img_float.max()
    else:
        max_value=strech_max
    min_value=img_float.min()
    dst=255*(img_float-min_value)/(max_value-min_value)
    dst=dst.astype(np.uint8)
    return dst
################################################################################3
def filter_area(conts_list:list,area_range:tuple=(100,500))->list:
    '''
    filter rec_area that is not in range,\n
    return conts_list, arrs_list
    '''
    conts_out=[]
    
    for i in range(len(conts_list)):
        rec_cont=cv2.minAreaRect(conts_list[i])
        rec_cont=cv2.boxPoints(rec_cont)
        rec_cont=np.int0(rec_cont)
        rec_area=cv2.contourArea(rec_cont)
        rec_area=abs(rec_area)
        
        if area_range[0]<rec_area<area_range[1]:
            
            conts_out.append(conts_list[i])
        
    return conts_out

def filter_nohavechild(conts_list:list,arrs_list:list)->tuple:
    '''filter conts which does not have a child_cont,leave the outest cont,notice that these outest do have child'''
    conts_out=[]
    arrs_out=[[]]
    for i in range(len(arrs_list[0])):
        #if have child cont
        if arrs_list[0][i][2]!=-1:
            conts_out.append(conts_list[i])
            arrs_out[0].append(arrs_list[0][i])
    return conts_out,arrs_out

def filter_havechild(conts_list:list,arrs_list:list)->tuple:
    '''filter conts which has a child_cont, leave the innerest cont,notice that these innerests may not have parent'''
    conts_out=[]
    arrs_out=[[]]
    for i in range(len(arrs_list[0])):
        #if do not have child cont
        if arrs_list[0][i][2]==-1:
            conts_out.append(conts_list[i])
            arrs_out[0].append(arrs_list[0][i])
    return conts_out,arrs_out

def find_original_num(ori_list:list,conts:np.ndarray)->int:
    '''return the nums in ori_conts_list'''
    for i in range(len(ori_list)):
        if ori_list[i].shape==conts.shape:
            if (ori_list[i]==conts).all():
                return i
#false
def find_current_num(cur_list:list,conts:np.ndarray)->int:
    '''return the nums in cur_conts_list'''
    for i in range(len(cur_list)):
        if i.shape==conts.shape:
            if (i==conts).all():
                return cur_list.index(i) 
       
   
    
def filter_strange(conts_list:list,ratio:float=5)->list:
    '''
    filter the shape looks so strange that their rec is too thin
    '''
    conts_out=[]

    for i in range(len(conts_list)):
        info_tuple=getrec_info(conts_list[i])
        x=info_tuple[0]
        y=info_tuple[1]
        wid=info_tuple[2]
        hei=info_tuple[3]
        proportion=wid/hei
        if proportion>1:
            if proportion>ratio:
                continue
        else:
            if 1/proportion>ratio:
                continue
        
        conts_out.append(conts_list[i])
    return conts_out

def filter_normal(conts_list,ratio:float=1.5)->list:
    '''
    filter the shape looks too normal that their rec is nearly square 
    '''
    conts_out=[]

    for i in range(len(conts_list)):
        x,y,wid,hei,_,_=getrec_info(conts_list[i])
        proportion=wid/hei
        if proportion>1:
            if proportion<ratio:
                continue
        else:
            if 1/proportion<ratio:
                continue
        
        conts_out.append(conts_list[i])

    return conts_out

def filter_no_shapelike(conts_list,area_near_range:tuple=(0.5,1.5),ratio_near_range:tuple=(0.5,1.5))->list:
    '''
    filter conts that has no shapelike cont ,\n
    ratio_range describes what is near;\n
    return list of tuple, each tuple is a pair of cont!!!
    '''
    
    conts_out=[]
    tuple_list=[]
    
    for i in range(len(conts_list)):
        center_x,center_y,wid,hei,rec_points,rec_area=getrec_info(conts_list[i])
        for j in tuple_list:
            #first rec_area is near
            if area_near_range[0]<rec_area/j[0]<area_near_range[1]:
                #second wid/hei is near:
                if ratio_near_range[0]<wid/hei/j[1]<ratio_near_range[1]:
                    conts_out.append((j[2],conts_list[i]))
        tuple_list.append((rec_area,wid/hei,conts_list[i]))
    return conts_out



def set_grades(conts_list:list)->dict:
    grades_dict={i:0 for i in range(len(conts_list))}
    return grades_dict

    
def color_test(conts_list:list,
               grades_dict:dict,
               ori_bin:np.ndarray,
               white_ratio_range:list=[0.6,1],
               center_wid:int=4):
    '''@brief access by white_ratio_range and center white judge
    white_ratio=white/whole'''
    for each in range(len(conts_list)):
        #create minrec,then calculate the color
        center_x,center_y,wid,hei,rec_points,_=getrec_info(conts_list[each])
        leftup_x=rec_points[0][0]
        leftup_y=rec_points[0][1]
        leftdown_x=rec_points[3][0]
        leftdown_y=rec_points[3][1]
        if leftdown_x==leftup_x:
            slope=0
        else:
            slope=(leftdown_y-leftup_y)/(leftdown_x-leftup_x)
        white_sum=0
        black_sum=0
        for i in range(int(hei)):
            if leftup_y+i>1023:
                break
            for j in range(int(wid)):
                if round(leftup_x+slope*i+j)>1279 or round(leftup_x+slope*i+j)<0:
                    break
                if ori_bin[leftup_y+i][round(leftup_x+slope*i)+j]==255:
                    white_sum+=1
                else:
                    black_sum+=1      
        #create center square,then calculate the color
        sum_center=0
        for i in range(center_wid):
            if center_y-center_wid/2+i<0 or center_y-center_wid/2+i>1023:
                break
            for j in range(center_wid):
                if center_x-center_wid/2+j<0 or center_x-center_wid/2+j>1279:
                    break
                sum_center+=ori_bin[round(center_y-center_wid/2+i)][round(center_x-center_wid/2+j)]
        area_center=center_wid*center_wid
        
        #begin access grade
        #white_ratio +10
        white_ratio=white_sum/(white_sum+black_sum)
        if white_ratio>white_ratio_range[0] and white_ratio<white_ratio_range[1]:
            grades_dict[each]+=10
        #white_center +5
        if sum_center>area_center*255*0.8:
            grades_dict[each]+=5

def parent_test(conts_list:list,
                arrs_list:list,
                grade_dict:dict,
                ori_cont_list:list,
                area_ratio:float=1.2):
    '''area_ratio=parent_area/child_area'''
    for i in range(len(arrs_list[0])):
        #if have parent:non solid always have parent
        if arrs_list[0][i][3]!=-1:
            parent_ori_num=arrs_list[0][i][3]
            parent_cont=ori_cont_list[parent_ori_num]
            child_cont=conts_list[i]
            
            parent_area=cv2.contourArea(parent_cont)
            child_area=cv2.contourArea(child_cont)
            #inner-outer match +10
            if parent_area<child_area*area_ratio and parent_area>child_area:
                grade_dict[i]+=10

def child_test(conts_list:list,
                arrs_list:list,
                grade_dict:dict,
                ori_cont_list:list,
                area_ratio:float=1.2):
    '''area_ratio=parent_area/child_area'''
    for i in range(len(arrs_list[0])):
        #if have child: enclosed shape usually have child
        if arrs_list[0][i][2]!=-1:
            child_ori_num=arrs_list[0][i][2]
            child_cont=ori_cont_list[child_ori_num]
            parent_cont=conts_list[i]
            
            parent_area=cv2.contourArea(parent_cont)
            child_area=cv2.contourArea(child_cont)
            #inner-outer match +10
            if parent_area<child_area*area_ratio and parent_area>child_area:
                grade_dict[i]+=10

def goodshape_test(conts_list:list,
                   grade_dict:dict,
                   area_ratio:float=0.7):
    '''area_ratio=cont_area/minrec_area'''
    for i in range(len(conts_list)):
        cont_area=cv2.contourArea(conts_list[i])
        rec_area=getrec_info(conts_list[i])[5]
        #good shape +10
        if cont_area/rec_area>area_ratio:
            grade_dict[i]+=10

def search_inner(ori_conts_list:list,
                 ori_arrs_list:list,
                 ori_bin:np.ndarray,
                 out_nums:int=3)->list:
    '''search for the best inner conts,return out_nums conts in list ,
    you may need to change some params to adapt to different inner shapes,
    actually return innerest conts'''
    if len(ori_conts_list)<=3:
        return ori_conts_list
    bigger_conts,bigger_arrs=filter_area(ori_conts_list,ori_arrs_list,200)
    if len(bigger_conts)<=3:
        return bigger_conts
    inner_conts,inner_arrs=filter_havechild(bigger_conts,bigger_arrs)
    if len(inner_conts)<=3:
        return inner_conts
    normal_conts,normal_arrs=filter_strange(inner_conts,inner_arrs,5)
    if len(normal_conts)<=3:
        return normal_conts
    grades_dict=set_grades(normal_conts)
    #you cannot filter normal_conts cause grade_dict is set on nums of range(len(normal_conts))
    color_test(normal_conts,grades_dict,ori_bin)
    parent_test(normal_conts,normal_arrs,grades_dict,ori_conts_list)
    goodshape_test(normal_conts,grades_dict)
    out_list=[]
    key_list=list(grades_dict.keys())
    value_list=list(grades_dict.values())
    for i in range(out_nums):
        max_index=value_list.index(max(value_list))
        out_list.append(normal_conts[key_list[max_index]])
        key_list.remove(key_list[max_index])
        value_list.remove(value_list[max_index])
    return out_list
        
def search_outer(ori_conts_list:list,
                 ori_arrs_list:list,
                 ori_bin:np.ndarray,
                 out_nums:int=3)->list:
    '''search for the best outer conts, return out_nums conts in list,
    actually return outest conts'''
    if len(ori_conts_list)<=3:
        return ori_conts_list
    bigger_conts,bigger_arrs=filter_area(ori_conts_list,ori_arrs_list,1500)
    if len(bigger_conts)<=3:
        return bigger_conts
    outer_conts,outer_arrs=filter_nohavechild(bigger_conts,bigger_arrs)
    if len(outer_conts)<=3:
        return outer_conts
    normal_conts,normal_arrs=filter_strange(outer_conts,outer_arrs,4)
    if len(normal_conts)<=3:
        return normal_conts
    grades_dict=set_grades(normal_conts)
    color_test(normal_conts,grades_dict,ori_bin,[0.05,0.4])
    child_test(normal_conts,normal_arrs,grades_dict,ori_conts_list,1.1)
    goodshape_test(normal_conts,grades_dict,0.9)
    out_list=[]
    key_list=list(grades_dict.keys())
    value_list=list(grades_dict.values())
    for i in range(out_nums):
        max_index=value_list.index(max(value_list))
        out_list.append(normal_conts[key_list[max_index]])
        key_list.remove(key_list[max_index])
        value_list.remove(value_list[max_index])
    return out_list

def isrelative(cont1:np.ndarray,
               cont2:np.ndarray,
               ori_conts_list:list,
               ori_arrs_list:list)->int:
    '''return 0 for none, 1 for cont1 is child, 2 for cont1 is parent'''
    ori_num1=find_original_num(ori_conts_list,cont1)
    ori_num2=find_original_num(ori_conts_list,cont2)
    if ori_num1==ori_arrs_list[0][ori_num2][2] or ori_num2==ori_arrs_list[0][ori_num1][3]:
        return 1
    elif ori_num1==ori_arrs_list[0][ori_num2][3] or ori_num2==ori_arrs_list[0][ori_num1][2]:
        return 2
    else:
        return 0

def iscenternear(cont1:np.ndarray,
                 cont2:np.ndarray,
                 distance:int=50)->bool:
    '''
    <distance return true\n
    else return false
    '''
    x1,y1,_,_,_,_=getrec_info(cont1)
    x2,y2,_,_,_,_=getrec_info(cont2)
    dis=((x1-x2)**2+(y1-y2)**2)**0.5
    if dis<distance:
        return True
    else:
        return False

def is_center_dis_good(cont1:np.ndarray,
                       cont2:np.ndarray,
                       center_dis_range:tuple):
    '''
    return True if center_dis is in range
    '''
    x1,y1,_,_,_,_=getrec_info(cont1)
    x2,y2,_,_,_,_=getrec_info(cont2)
    dis=((x1-x2)**2+(y1-y2)**2)**0.5
    if center_dis_range[0]<dis<center_dis_range[1]:
        return True
    else:
        return False
    
    

def walk_until_white(begin_x:int,
                     begin_y:int,
                     slope:float,
                     img_edge_bin:np.ndarray,
                     direction:int=0)->tuple:
    '''direction=0 for x plus 1 direction, =1 for x minus dirction
    ,   return x,y'''
    x=begin_x
    y=begin_y
    if direction==0:
        while True:
            x+=1
            y+=slope
            if x>1279 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return min(x,1279),y
                elif y<0:
                    y=0
                    return min(x,1279),y
                elif x>1279:
                    x=1279
                    return x,y
            if img_edge_bin[round(y)][x]==255:
                return x,round(y)
    if direction==1:
        while True:
            x-=1
            y+=slope
            if x<0 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return max(x,0),y
                elif y<0:
                    y=0
                    return max(x,0),y
                elif x<0:
                    x=0
                    return x,y
            if img_edge_bin[round(y)][x]==255:
                return x,round(y) 

 
def walk_until_black(begin_x:int,begin_y:int,slope:float,img_edge_bin:np.ndarray,
                     direction:int=0)->tuple:
    '''direction =0 for x+=1, or 1 for x-=1'''      
    x=begin_x
    y=begin_y
    if direction==0:
        while True:
            x+=1
            y+=slope
            if x>1279 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return min(x,1279),y
                elif y<0:
                    y=0
                    return min(x,1279),y
                elif x>1279:
                    x=1279
                    return x,y
            if img_edge_bin[round(y)][x]==0:
                return x,round(y)
    if direction==1:
        while True:
            x-=1
            y+=slope
            if x<0 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return max(x,0),y
                elif y<0:
                    y=0
                    return max(x,0),y
                elif x<0:
                    x=0
                    return x,y
            if img_edge_bin[round(y)][x]==0:
                return x,round(y)

def walk_until_dis(begin_x:int,
                   begin_y:int,
                   slope:float,
                   distance:float,
                   direction:str='right',
                   img_size_yx:tuple=(1024,1280)
                   
                    )->list:
    '''direction =right for x+=1, or left for x-=1,
    return x,y'''      
    x=begin_x
    y=begin_y
    x_range=(0,img_size_yx[1])
    y_range=(0,img_size_yx[0])
    if math.isinf(slope) or math.isnan(slope):
        print('slope out')
        return [x,y]
    theta=math.atan(abs(slope))
    delta_x=distance*math.cos(theta)
    
    delta_y=distance*math.sin(theta)
    if direction=='right':
        
        x+=delta_x
        y=y+delta_y if slope>0 else y-delta_y
        #check range 
        if x>x_range[1]:
            x=x_range[1]
        if y>y_range[1]:
            y=y_range[1]
        if y<y_range[0]:
            y=y_range[0]
        return [x,round(y)]
    if direction=='left':
        x-=delta_x
        y=y+delta_y if slope<0 else y-delta_y
        #check range
        if x<x_range[0]:
            x=x_range[0]
        if y>y_range[1]:
            y=y_range[1]
        if y<y_range[0]:
            y=y_range[0]
        return [x,round(y)]

def make_plate(cont:np.ndarray,img_edge_bin:np.ndarray)->np.ndarray:
    '''make the plate yourself by innercont, notice that img_edge_bin has to be dealed with medianblur, then binary'''
    
    center_x,center_y,wid,hei,corners,_=getrec_info(cont)
    x1,y1=corners[0]
    x2,y2=corners[1]
    x3,y3=corners[2]
    x4,y4=corners[3]
    slope12=(y2-y1)/(x2-x1)
    slope12_=(-1)/slope12
    x12_=round((x1+x2)/2)
    y12_=round((y1+y2)/2)
    x23_=round((x2+x3)/2)
    y23_=round((y2+y3)/2)
    x34_=round((x3+x4)/2)
    y34_=round((y3+y4)/2)
    x14_=round((x1+x4)/2)
    y14_=round((y1+y4)/2)
    #get the lengths to be lengthen
    x12,y12=walk_until_black(x12_,y12_,slope12_,img_edge_bin,0)
    x12,y12=walk_until_white(x12,y12,slope12_,img_edge_bin,0)
    x23,y23=walk_until_black(x23_,y23_,slope12,img_edge_bin,0)
    x23,y23=walk_until_white(x23,y23,slope12,img_edge_bin,0)
    x34,y34=walk_until_black(x34_,y34_,slope12_,img_edge_bin,1)
    x34,y34=walk_until_white(x34,y34,slope12_,img_edge_bin,1)
    x14,y14=walk_until_black(x14_,y14_,slope12,img_edge_bin,1)
    x14,y14=walk_until_white(x14,y14,slope12,img_edge_bin,1)
    
    points=[[x12,y12],[x23,y23],[x34,y34],[x14,y14]]
    begins=[[x12_,y12_],[x23_,y23_],[x34_,y34_],[x14_,y14_]]
    dis_list=[0,0,0,0]
    for i in range(len(points)):
        if i!=[0,0]:
            dis_list[i]=((points[i][0]-begins[i][0])**2+(points[i][1]-begins[i][1])**2)**0.5
    #begin to access which side to lengthen
    grade12=0
    grade23=0
    #compare uses as a good range
    compare=(wid+hei)/2
    if dis_list[0]!=0 and dis_list[2]!=0:
        #if lengths ro be lengthen are match,grade+10
        if (dis_list[0]-dis_list[2])**2<225:
            grade12+=10
        #if lengths to be lengthen are in a good range,grade+5
        if dis_list[0]<compare and dis_list[2]<compare:
            grade12+=5
        #if both out of range,grade-10
        if dis_list[0]>compare*2.5 and dis_list[2]>compare*2.5:
            grade12-=10
        #if a side out of range,grade-5
        elif dis_list[0]>compare*2.5 or dis_list[2]>compare*2.5:
            grade12-=5
    if dis_list[1]!=0 and dis_list[3]!=0:
        if (dis_list[1]-dis_list[3])**2<225:
            grade23+=10
        if dis_list[1]<compare and dis_list[3]<compare:
            grade23+=5
        if dis_list[1]>compare*2.5 and dis_list[3]>compare*2.5:
            grade23-=10
        elif dis_list[1]>compare*2.5 or dis_list[3]>compare*2.5:
            grade23-=5
    #if 23 side should be lengthen
    if grade12>grade23:
        if grade12>5:
            meandis=(dis_list[0]+dis_list[2])/2
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12_,meandis,0)
            out_points[1]=walk_until_dis(x2,y2,slope12_,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12_,meandis,1)
            out_points[3]=walk_until_dis(x4,y4,slope12_,meandis,1)
            out_points=np.array(out_points)
            return out_points
        else:
            meandis=compare
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12_,meandis,0)
            out_points[1]=walk_until_dis(x2,y2,slope12_,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12_,meandis,1)
            out_points[3]=walk_until_dis(x4,y4,slope12_,meandis,1)
            out_points=np.array(out_points)
            return out_points
    #if 12 side should be lengthen
    elif grade12<grade23:
        if grade23>5:
            meandis=(dis_list[1]+dis_list[3])/2
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12,meandis,1)
            out_points[1]=walk_until_dis(x2,y2,slope12,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12,meandis,0)
            out_points[3]=walk_until_dis(x4,y4,slope12,meandis,1)
            out_points=np.array(out_points)
            return out_points
        else:
            meandis=compare
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12,meandis,1)
            out_points[1]=walk_until_dis(x2,y2,slope12,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12,meandis,0)
            out_points[3]=walk_until_dis(x4,y4,slope12,meandis,1)
            out_points=np.array(out_points)
            return out_points         
    #if fail to access, make a square
    else:
        meandis=compare/4
        out_points=[[],[],[],[]]
        slope13=(y1-y3)/(x1-x3)
        slope24=(y4-y2)/(x4-x2)
        out_points[0]=walk_until_dis(x1,y1,slope13,meandis,1)
        out_points[1]=walk_until_dis(x2,y2,slope24,meandis,0)
        out_points[2]=walk_until_dis(x3,y3,slope13,meandis,0)
        out_points[3]=walk_until_dis(x4,y4,slope24,meandis,1)
        out_points=np.array(out_points)
        return out_points

def make_big_rec(cont1:np.ndarray,
                 cont2:np.ndarray
                 )->tuple:
    '''
    vstack two conts into one cont and then getrec_info\n
    return tuple:  (center_x,center_y,wid,hei,rec_points,rec_area)
    
    '''
    big_cont=np.vstack((cont1,cont2))
    return getrec_info(big_cont)

def expand_rec_wid(rec_cont_list,expand_rate:float=1.5,img_size_yx:tuple=(1024,1280))->list:
    '''
    only expand short side
    return a list of expanded rec_conts
    '''
    out_list=[]
    dis=0
    for i in rec_cont_list:
        center_x,center_y,wid,hei,rec_points,_=getrec_info(i)
        out_points=[[],[],[],[]]
        x1,y1=rec_points[0]
        x2,y2=rec_points[1]
        x3,y3=rec_points[2]
        x4,y4=rec_points[3]
        hei=((x1-x2)**2+(y1-y2)**2)**0.5
        wid=((x2-x3)**2+(y2-y3)**2)**0.5
        if wid<hei:
            #always make side12 is short one
            wid,hei=hei,wid
            x1,x2,x3,x4=x2,x3,x4,x1
            y1,y2,y3,y4=y2,y3,y4,y1
            dis=hei*(expand_rate-1)/2
            #all is expanding in slope 12, the short side slope
            slope12=(y1-y2)/(x1-x2)
            #check if vertical
            if slope12==0 or math.isinf(slope12) or math.isnan(slope12):
                out_points[1]=[x1,y1-dis]
                out_points[2]=[x2,y2+dis]
                out_points[3]=[x3,y3+dis]
                out_points[0]=[x4,y4-dis]
                
            else:
                #1 and 4 is same, 1 and 2 is opposite
                
                        
                direc1='left' if slope12>0 else 'right'
                direc2='right' if slope12>0 else 'left'
                #make 1 is left down point
                out_points[1]=walk_until_dis(x1,y1,slope12,dis,direc1,img_size_yx=img_size_yx)
                out_points[2]=walk_until_dis(x2,y2,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[3]=walk_until_dis(x3,y3,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[0]=walk_until_dis(x4,y4,slope12,dis,direc1,img_size_yx=img_size_yx)
            out_list.append(np.array(out_points,dtype=np.int64))
        else:
            #always make side12 is short one
 
            dis=hei*(expand_rate-1)/2
            #all is expanding in slope 12, the short side slope
            
            slope12=(y1-y2)/(x1-x2)
            #check if vertical
            if slope12==0 or math.isnan(slope12) or math.isinf(slope12):
                out_points[0]=[x1,y1+dis]
                out_points[1]=[x2,y2-dis]
                out_points[2]=[x3,y3-dis]
                out_points[3]=[x4,y4+dis]
            else:
                #1 and 4 is same, 1 and 2 is opposite
                direc1='right' if slope12>0 else 'left'
                direc2='left' if slope12>0 else 'right'
                #make 1 is left down point
                out_points[0]=walk_until_dis(x1,y1,slope12,dis,direc1,img_size_yx=img_size_yx)
                out_points[1]=walk_until_dis(x2,y2,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[2]=walk_until_dis(x3,y3,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[3]=walk_until_dis(x4,y4,slope12,dis,direc1,img_size_yx=img_size_yx)
            out_list.append(np.array(out_points,dtype=np.int64))
    return out_list     
        
            
        
        


def auto_search(ori_img:np.ndarray)->tuple:
    '''return 4 points that is the best cont of plate,tuple[0]is auto, tuple[1] is handmade'''
    img_edge_bin=get_edgebin2(ori_img)
    ori_bin=image_binary(ori_img)
    ori_conts,ori_arrs=cv2.findContours(img_edge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    inner_conts=search_inner(ori_conts,ori_arrs,ori_bin)
    outer_conts=search_outer(ori_conts,ori_arrs,ori_bin)
    best_outer_num=0
    best_inner_num=0
    for i in range(len(inner_conts)):
        for j in range(len(outer_conts)):
            if isrelative(inner_conts[i],outer_conts[j],ori_conts,ori_arrs)==1:
                if iscenternear(inner_conts[i],outer_conts[j]):
                    best_outer_num=j
                    best_inner_num=i
    hand_points=make_plate(inner_conts[best_inner_num],img_edge_bin)
    auto_points=getrec_info(outer_conts[best_outer_num])[4]
    return auto_points,hand_points

def getdia_info(pt0,pt1,angle=0):
    '''input diagonal points of rec,\n
    return 
    cx\n
    cy\n
    w\n
    h\n
    rec\n
    area'''
    x0=pt0[0]
    x1=pt1[0]
    y0=pt0[1]
    y1=pt1[1]
    if angle==0:
        pt2=[x0,y1]
        pt3=[x1,y0]
        c_x,c_y,w,h,rec_cont,rec_area=getrec_info(np.array([pt0,pt1,pt2,pt3]))
        return c_x,c_y,w,h,rec_cont,rec_area



#***********************for yolov5************************#
def drawrec_and_getcenter(dia_list,ori_img,camera_center):
    '''return drawed_img, center\n
    find nearset camera face'''

    dis=100000
    recinfo_list=[]
    index=0
    img_copy=ori_img.copy()
    #search biggest area rec
    for i in range(len(dia_list)):
        
        recinfo_list.append(getdia_info(dia_list[i][0],dia_list[i][1]))
        cx=recinfo_list[i][0]
        cy=recinfo_list[i][1]
        
        center_dis=math.sqrt((cx-camera_center[0])**2+(cy-camera_center[1])**2)
        if center_dis<dis:
            dis=center_dis
            index=i
    add_text(img_copy,'nearest dis',dis,(0,100))
    final_center=(int(recinfo_list[index][0]),int(recinfo_list[index][1]))

    final_reccont=recinfo_list[index][4]
    
    
    
    cv2.drawContours(img_copy,[final_reccont],-1,color=(128,128,255))
    cv2.circle(img_copy,final_center,10,(255,128,128),-1)
    return img_copy,final_center



class PIDtrace:
    '''
    input must be np.ndarray
    input  current_vector(matrix is also ok) and dim_target\n
    output vector ,direction and size is calculated by pid
    '''
    def __init__(self,kp,ki,kd,shape) :
        self.kp=kp
        self.kd=kd
        self.ki=ki
        self.shape=shape
        self.error=np.zeros(shape)
        self.integral=np.zeros(shape)
        self.diff=np.zeros(shape)
        self.pre_error=np.zeros(shape)
    def update(self,act,exp):
        '''
        act=actual_value=star_location\n
        exp=expectation=target_location
        '''
        act,exp=check_and_change_shape(act,exp,self.shape)
        self.error=exp-act
        self.integral+=self.error
        self.diff=self.error-self.pre_error
        self.pre_error=self.error
        pid_value=self.kp*self.error+self.ki*self.integral+self.kd*self.diff
        return pid_value
    
def draw_pid_vector(img:np.ndarray,act,pid_value):
    act,pid_value=check_and_change_shape(act,pid_value,(2,1))
    start_point=(int(act[0]),int(act[1]))
    end_point=act+pid_value
    end_point=(int(end_point[0]),int(end_point[1]))
    cv2.arrowedLine(img,start_point,end_point,(128,255,128))
    return img
    
    


def check_and_change_shape(x,y,shape:tuple)->np.ndarray:
    '''input list or np.ndarray,return np.ndarray'''
    
    x=np.reshape(x,shape)
    y=np.reshape(y,shape)
    return x,y
    
       

def draw_time_correlated_value(img:np.ndarray,
                               timecount,
                               value,
                               value_scope:tuple,
                               point_radians:5,
                               point_color:tuple = (0,255,0))->np.ndarray:
    '''
    Warning: you need to clear img yourself if timecount overflow
    '''
    
    x = round(timecount)
    
    y = round(img.shape[0] + ((value - value_scope[0])/(value_scope[1]-value_scope[0])) * (-img.shape[0]))
    
    cv2.circle(img,(x,y),point_radians,point_color,-1)
    
    return img
 
 
 
def draw_crosshair(img:np.ndarray,color:tuple = (0,255,0),thickness :int =2, len_ratio:float = 0.1):
    hei = img.shape[0]
    wid = img.shape[1]

    
    
    x = round(wid/2)
    y = round(hei/2)
    crosshair_wid_length = round(len_ratio * wid)
    crosshair_hei_length = round(len_ratio * hei)

    cv2.circle(img,(x,y),round(0.01*min(wid,hei)),(0,0,255),-1)
    cv2.line(img,(x - crosshair_wid_length,y),(x + crosshair_wid_length,y),color,thickness)
    cv2.line(img,(x,y - crosshair_hei_length),(x,y + crosshair_hei_length),color,thickness)
    
    return img

def turn_big_rec_list_to_center_points_list(rec_list:list)->list:
    if rec_list is None:
        return None
    out = []
    for i in rec_list:
        x,y,_,_,_,_=getrec_info(i)
        out.append([round(x),round(y)])
    return out
  
############################################################IMG API FOR NETWORK###########################################

class Img:
    def __init__(self) -> None:
        pass
    
    class canvas:
        def __init__(self,size_tuple:tuple,color:str='white'):
            '''
            size_tuple: (width,height,channel)
            '''
            img=np.zeros(size_tuple,dtype=np.uint8)
            if color=='white':
                img.fill(255)
            self.color = color
            self.img=img
            self.wid=img.shape[1]
            self.hei=img.shape[0]
            self.timecount = 0
            self.timecount_max = self.wid - 1
            
            
        def draw_rec(self,color:tuple=(0,0,255),mode:int=0,info:Union[np.ndarray,tuple,None]=None,nums:int=1):
            '''
            color:(0,0,255) for red
            mode:   0 : center
                    1 : random
                    2 : specified
            info : only for specified mode: 
                tuple: ((center_x_int,center_y_int),(wid_int,hei_int),angle_int)
                np.ndarray: batched_boxpoints
                
            nums: nums of rec to draw, only for random mode
            '''
            if mode==0:
                center=(round(self.wid/2),round(self.hei/2))
                width=round(self.wid/6)
                height=round(self.hei/6)
                angle=30
                bp=cv2.boxPoints((center,(width,height),angle))
                Img.draw_rec(self.img,bp,color=color)
            elif mode==1:
                for i in range(nums):
                    center=(np.random.randint(0,self.wid),np.random.randint(0,self.hei))
                    width=np.random.randint(5,min(self.wid,self.hei)/4)
                    height=np.random.randint(5,min(self.wid,self.hei)/4)
                    angle=np.random.randint(1,90)
                    bp=cv2.boxPoints((center,(width,height),angle))
                    Img.draw_rec(self.img,bp,color=color)
            elif mode==2:
                if isinstance(info,tuple):
                    bp=Img.Change.toboxpoints(info)
                    
                elif isinstance(info,np.ndarray):
                    bp=info
                else:
                    raise TypeError('info must be tuple or np.ndarray')
                Img.draw_rec(self.img,bp,color=color)
            else:
                raise TypeError('mode is wrong')
        def show(self):
            '''
            Only for quick show, will block until pressing
            '''
            cv2.imshow('canvas',self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        
        def addtxt(self,
                   name:str,
                   value,
                   pos:tuple,
                   color:tuple = (100,200,200),
                   scale_size:float = 0.5
                   ):
            self.img = add_text(self.img,
                                name=name,
                                value=value,
                                pos=pos,
                                color=color,
                                scale_size=scale_size
                                )    
            
        def draw_time_correlated_value(self,value,value_scope:tuple,point_radians:5,point_color:tuple = (0,255,0)):
            self.timecount+=1
            
            x = round(self.timecount)
            
            y = round(self.hei + ((value - value_scope[0])/(value_scope[1]-value_scope[0])) * (-self.hei))
            
            cv2.circle(self.img,(x,y),point_radians,point_color,-1)
            
            if self.timecount == self.timecount_max:
                self.timecount = 0
                
                self.img=np.zeros((self.wid,self.hei),dtype=np.uint8)
                if self.color=='white':
                    self.img.fill(255)
                
            
                
        
        
    class Check:
        def __init__(self) -> None:
            pass
        @classmethod
        def is_3channelimg(cls,img:np.ndarray):
            '''
            input cannot be batch_img\n
            return Trun if img is 3 channel\n
            return False if img is grayscale
            '''
            if len(img.shape)==3:
                return True if img.shape[2]==3 else False
            elif len(img.shape)==2:
                return False
            else:
                raise TypeError("img.shape is not 3 or 2 length")
        @classmethod
        def is_batch_contour(cls,mat_like:np.ndarray):
            return True if len(mat_like.shape)==3 else False
        @classmethod
        def is_batch_img(cls,mat_like:np.ndarray):
            return True if len(mat_like.shape)==4 else False
                
    @classmethod
    def draw_rec(cls,
                 ori_img:np.ndarray,
                 box_points_batch:np.ndarray,
                 sequence:Optional[np.ndarray]=None,
                 color:tuple=(127,127,255),
                 thickness:int=-1
                 ):
        """draw a series of rec or a single rec on ori_img.Will change dtype to int32 of box_points

        Args:
            ori_img (np.ndarray): WILL CHANGE ORI_IMG, COPY BEFORE ENTTER
            box_points (np.ndarray): 3 dim array, first dim is batch_size
            sequence (np.ndarray | None, optional): the index array of rec in box_points you want to show, Defaults to None.
            color (tuple, optional): tuple of color Defaults to (127,127,255).
            thickness (int, optional): -1 is solid, other is width. Defaults to -1.
        """
        
        if not cls.Check.is_3channelimg(ori_img):
            ori_img=cv2.cvtColor(ori_img,cv2.COLOR_GRAY2BGR)
        if not cls.Check.is_batch_contour(box_points_batch):
            box_points_batch=np.expand_dims(box_points_batch,axis=0)
        #!!!MUST BE INT OE NP.INT32
        box_points_batch=np.round(box_points_batch).astype(np.int32)
        if sequence==None:
            cv2.drawContours(ori_img,box_points_batch,-1,color,thickness)
        else:
            cv2.drawContours(ori_img,box_points_batch[sequence],-1,color,thickness)
        return ori_img
    @classmethod
    def normalize(cls,ori_img:np.ndarray,scope:tuple=(0,1)):
        '''(y-a)/(b-a)=(x-xmin)/(xmax-xmin)'''
        return (ori_img-ori_img.min())/(ori_img.max()-ori_img.min())*(scope[1]-scope[0])+scope[0]
 
    class Change:
        def __init__(self) -> None:
            pass
        @classmethod
        def toimgbatch(cls,*args:np.ndarray):
            '''
            change to 4 dim, (batchsize,channel,wid,height)
            
            args=img1,img2...
            '''
            for i in args:
                if len(i.shape)==2:
                    i=np.expand_dims(i,0)
                else:
                    i=i.transpose(2,0,1)
            return np.stack(args,axis=0)
        
        @classmethod
        def toboxpoints(cls,info:tuple):
            '''
            tuple=(center,(wid,hei),angle)
            '''
            
            center=(round(info[0][0]),round(info[0][1]))
            wid=round(info[1][0])
            hei=round(info[1][1])
            angle=round(info[2])
            return cv2.boxPoints((center,(wid,hei),angle)).astype(np.int32)
        @classmethod
        def order_rec_points(cls,rec_points:np.ndarray)->np.ndarray:
            '''
            return rec_points in order:\n
            up_left\n
            up_right\n
            bottom_right\n
            bottom_left\n
            '''
            ordered_points = np.zeros_like(rec_points)

            #find up_left and bottom_right
            #sum=x+y
            sums = rec_points.sum(axis=1)
            ordered_points[0] = rec_points[np.argmin(sums)]
            ordered_points[2] = rec_points[np.argmax(sums)]

            #find right_up and bottom_left
            #diff=y-x
            diffs = np.diff(rec_points, axis=1)
            #smallest is up_right
            ordered_points[1] = rec_points[np.argmin(diffs)]
            #biggest is bottom_left
            ordered_points[3] = rec_points[np.argmax(diffs)]

            return ordered_points 
        
        @classmethod 
        def compress_img(cls,img:np.ndarray, k:int=50):
            '''
            return svd 
            '''
            

            compressed_channels = []
            for channel in cv2.split(img):
                U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                U_k = U[:, :k]
                S_k = np.diag(S[:k])
                Vt_k = Vt[:k, :]
                compressed_channel = U_k@S_k@Vt_k
                compressed_channel = np.round(compressed_channel).astype(np.uint8)
                compressed_channels.append(compressed_channel)

            compressed_image = cv2.merge(compressed_channels)
            return compressed_image

        
def trackbar_init(name,range:tuple,window:str='config'):
    def call(x):
        pass
    cv2.namedWindow(window,cv2.WINDOW_FREERATIO)
    cv2.createTrackbar(name,window,range[0],range[1],call)
    



    
    
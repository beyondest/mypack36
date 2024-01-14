import cv2
import time
import numpy as np
from typing import Union,Optional
import math

def order_rec_points(rec_points:np.ndarray)->np.ndarray:
        '''
        return rec_points in order:\n
        up_left\n
        up_right\n
        bottom_right\n
        bottom_left\n
        Notice : will reshape to (4,2) if input is (4,1,2)
        '''
        if len(rec_points.shape) == 3:
            rec_points_copy = rec_points.reshape((4,2))
        else:
            rec_points_copy = rec_points
        ordered_points = np.zeros_like(rec_points_copy)

        #find up_left and bottom_right
        #sum=x+y
        sums = rec_points_copy.sum(axis=1)
        ordered_points[0] = rec_points_copy[np.argmin(sums)]
        ordered_points[2] = rec_points_copy[np.argmax(sums)]

        #find right_up and bottom_left
        #diff=y-x
        diffs = np.diff(rec_points_copy, axis=1)
        #smallest is up_right
        ordered_points[1] = rec_points_copy[np.argmin(diffs)]
        #biggest is bottom_left
        ordered_points[3] = rec_points_copy[np.argmax(diffs)]

        return ordered_points 
    
    
def make_big_rec(cont1:np.ndarray,
                 cont2:np.ndarray
                 )->np.ndarray:
    '''
    vstack two conts into one cont and then getrec_info\n
    return tuple:  (center_x,center_y,wid,hei,rec_points,rec_area)
    
    '''
    big_cont=np.vstack((cont1,cont2))
    return getrec_info(big_cont)[4]


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
    if len(cont) != 4:
        ret=cv2.minAreaRect(cont)
        right_order_rec=cv2.boxPoints(ret)
        right_order_rec = order_rec_points(right_order_rec)
    else:

        right_order_rec = order_rec_points(cont)
    
    ret = cv2.minAreaRect(right_order_rec)
    bo = cv2.boxPoints(ret)
    bo=np.int0(bo)
   # rec_area=cv2.contourArea(bo)
   # rec_area=abs(rec_area)
    
    width = np.linalg.norm(right_order_rec[1] - right_order_rec[0])
    height = np.linalg.norm(right_order_rec[3] - right_order_rec[0])
    rec_area = width * height
    
    
    return (ret[0][0],ret[0][1],width,height,bo,rec_area)


def gray_stretch(image_gray:np.ndarray,strech_max:Optional[int]=None):
    
    img_float=image_gray.astype(np.float32)
    if strech_max==None:
        
        max_value=img_float.max()
    else:
        max_value=strech_max
        
        
    min_value=img_float.min()
    dst=255*(img_float-min_value)/(max_value-min_value)
    dst=dst.astype(np.uint8)
    return dst


def turn_big_rec_list_to_center_points_list(rec_list:list)->list:
    if rec_list is None:
        return None
    out = []
    for i in rec_list:
        x,y,_,_,_,_=getrec_info(i)
        out.append([round(x),round(y)])
    return out


def draw_single_cont(ori_copy:np.ndarray,cont:np.ndarray,color:int=2,thick:int=2):
    '''
    @cont:  np.array(left_down,left_up,right_up,right_down)
    @color=0,1,2=blue,green,red
    @thick:   defalt is 2
    '''
    colorlist=[(255,0,0),(0,255,0),(0,0,255)]
    cv2.drawContours(ori_copy,[cont],-1,colorlist[color],thick)
    return 0


def draw_big_rec_list(big_rec_list,img_bgr:np.ndarray,if_draw_on_input:bool = True)->np.ndarray:
    '''
    return img_bgr
    '''
    if big_rec_list is None:
        return None
    
    if if_draw_on_input:

        for i in big_rec_list:
            draw_single_cont(img_bgr,i,2,3)
            return img_bgr
    else:
        for i in big_rec_list:
            out = img_bgr.copy()
            draw_single_cont(img_bgr,i,2,3)
            return out
            
    

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
        out = img_bgr.copy()
        for i in center_list:
            cv2.circle(out,center_list,radius,color,-1)
            
        return out

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

def make_big_rec(cont1:np.ndarray,
                 cont2:np.ndarray
                 )->tuple:
    '''
    vstack two conts into one cont and then getrec_info\n
    return tuple:  (center_x,center_y,wid,hei,rec_points,rec_area)
    
    '''
    big_cont=np.vstack((cont1,cont2))
    return getrec_info(big_cont)[4]

def expand_rec_wid(rec_cont_list:Union[list,None],expand_rate:float=1.5,img_size_yx:tuple=(1024,1280))->Union[list,None]:
    '''
    only expand short side
    return a list of expanded rec_conts
    '''
    if rec_cont_list is None:
        return None
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
        

def cvshow(img:np.ndarray,windows_name:str='show'):
    '''
    use to show quickly
    '''
    cv2.imshow(windows_name,img)
    while True:
        if (cv2.waitKey(0) & 0xff )== 27:
            break
    cv2.destroyAllWindows()
    
def normalize(ori_img:np.ndarray,scope:tuple=(0,1))->np.ndarray:
    '''(y-a)/(b-a)=(x-xmin)/(xmax-xmin)'''
    return (ori_img-ori_img.min())/(ori_img.max()-ori_img.min())*(scope[1]-scope[0])+scope[0]

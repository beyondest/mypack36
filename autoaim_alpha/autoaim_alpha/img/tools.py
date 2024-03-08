
from ..os_op.basic import *
import numpy as np
import cv2
from typing import Union,Optional
import math
import matplotlib.pyplot as plt

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
        
        
    def show(self):
        '''
        Only for quick show, will block until pressing
        '''
        cv2.imshow('canvas',self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    if max_value - min_value == 0:
        return image_gray
    
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


def draw_single_cont(ori_copy:np.ndarray,cont:np.ndarray,color:tuple=(0,255,0),thick:int=2):
    '''
    @cont:  np.array(left_down,left_up,right_up,right_down)
    @color=0,1,2=blue,green,red
    @thick:   defalt is 2
    '''
    cv2.drawContours(ori_copy,[cont],-1,color,thick)
    return 0


def draw_big_rec_list(big_rec_list,
                      img_bgr:np.ndarray,
                      if_draw_on_input:bool = True,
                      color:tuple = (0,255,0))->np.ndarray:
    '''
    return img_bgr
    '''
    if big_rec_list is None:
        return None
    
    if if_draw_on_input:
        for i in big_rec_list:
            draw_single_cont(img_bgr,i,color)
        return img_bgr
    
    else:
        out = img_bgr.copy()
        for i in big_rec_list:
            draw_single_cont(img_bgr,i,color=color)
        return out
        
            
    

def draw_center_list(center_list:list,
                     img_bgr:np.ndarray,
                     if_draw_on_input:bool = True,
                     color = (0,255,0)
                     ) ->np.ndarray:
    if center_list is None:
        return None
    
    radius = round((img_bgr.shape[0]+img_bgr.shape[1])/200)
    
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
             pos=None,
             font:int=cv2.FONT_HERSHEY_SIMPLEX,
             color:tuple=(0,0,0),
             scale_size: float= 0.5
             )->np.ndarray:
    
    '''
    show name:value on the position of pos(x,y)\n
    if pos is None, then auto position\n
    return img_bgr
    '''
    
    img_size_yx=(img_bgr.shape[0],img_bgr.shape[1])
    
    if pos is None:
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
        lr1.debug('Img : walk_until_dis fail,slope out')
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
            slope12=(y1-y2)/(x1-x2) if x1 - x2 != 0 else 0
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
            
            slope12=(y1-y2)/(x1-x2) if x1 - x2 != 0 else 0
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
    if ori_img is None:
        return None
    '''(y-a)/(b-a)=(x-xmin)/(xmax-xmin)'''
    divide_value=ori_img.max()-ori_img.min()
    
    if divide_value == 0:
        return np.zeros_like(ori_img)
    else:
        return (ori_img-ori_img.min())/divide_value*(scope[1]-scope[0])+scope[0]



def findpeaks(data, spacing=1, limit=None):
    """Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param data: values
    :param spacing: minimum spacing to the next peak (should be 1 or more)
    :param limit: peaks should have value greater or equal
    :return:
    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind




def get_threshold(img_for_show_in_hist:np.ndarray,
                  max_value:int=255,
                  mode:str='Dbg'):
    
    '''if mode == 'Dbg':
        d = cv2.resize(img_for_show_in_hist,(300,300))
        cv2.imshow('cal hist img',d)
        cv2.waitKey(1)'''
        
    x_axis_length  = 255
    
    hist = cv2.calcHist([img_for_show_in_hist],[0],None,[x_axis_length],[0,255])
    
    hist = normalize(hist,scope=(0,max_value))
    hist = np.round(hist).astype(np.int)
    thresh = np.argmax(hist)
    
    if thresh == len(hist) - 1:
        pass
    
    elif hist[thresh+1].item() < 30:
        thresh +=1
    
    if mode == 'Dbg':
        canvas_wid = 300
        canvas_hei = 300
        canvas = np.ones((canvas_wid,canvas_hei,3),dtype=np.uint8) * 255
       
        if hist.size == 0:
            lr1.error(f'hist size is 0 ,hist {hist}')
            return thresh
        non_zero_max_index = np.argwhere(hist > 0).max()
        if non_zero_max_index == 0:
            non_zero_max_index = 1
        step = canvas_wid // non_zero_max_index
        for i in range(non_zero_max_index):
            if i == non_zero_max_index - 1:
                break
            x0,y0,x1,y1 =  round(i*step), round(canvas_hei - hist[i].item()) ,round((i+1)*step) , round(canvas_hei - hist[i+1].item()) 
            cv2.line(canvas,(x0,y0),(x1,y1), color=(255,0,0))
            add_text(canvas,'',i,(x0,y0),color=(0,0,255),scale_size=0.8)
            
        add_text(canvas,'thresh',thresh,scale_size=0.8)
        
        cv2.imshow('hist',canvas)
        cv2.waitKey(1)

        lr1.debug(f'thresh:{thresh}')
    
    
    return thresh


def get_trapezoid_corners(rect1, rect2):
    ordered_pts = np.zeros((4, 2))
    
    rect1 = cv2.minAreaRect(rect1)
    rect2 = cv2.minAreaRect(rect2)
    rect1 = np.int0(cv2.boxPoints(rect1))
    rect2 = np.int0(cv2.boxPoints(rect2))

    if getrec_info(rect1)[0] < getrec_info(rect2)[0]:
        
        ordered_rect1 = order_rec_points(rect1)
        ordered_rect2 = order_rec_points(rect2)
    else:
        ordered_rect1 = order_rec_points(rect2)
        ordered_rect2 = order_rec_points(rect1)
    
        
    ordered_pts[0] = ordered_rect1[0]
    ordered_pts[1] = ordered_rect2[1]
    ordered_pts[2] = ordered_rect2[2]
    ordered_pts[3] = ordered_rect1[3]
    
    return ordered_pts


def CAL_EUCLIDEAN_DISTANCE(pt1,pt2)->float:
    
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5


def expand_trapezoid_wid(trapezoid_cont_list:Union[list,None],expand_rate:float=1.5,img_size_yx:tuple=(1024,1280)):
    if trapezoid_cont_list is None:
        return None
    
    
    out_list=[]
    for i in trapezoid_cont_list:
        expanded_trapezoid_points = np.zeros((4, 2))
        dis_left = CAL_EUCLIDEAN_DISTANCE(i[0],i[3]) * expand_rate / 2
        dis_right = CAL_EUCLIDEAN_DISTANCE(i[1],i[2]) * expand_rate / 2
        center_left = (i[0] + i[3])/2
        center_right = (i[1] + i[2])/2
        
        if i[0][0] - i[3][0] == 0:
            expanded_trapezoid_points[0] = center_left - dis_left * np.array([0,1])
            expanded_trapezoid_points[3] = center_left + dis_left * np.array([0,1])
        
        else:
            slope_left = (i[0][1] - i[3][1])/(i[0][0] - i[3][0])
            
            expanded_trapezoid_points[0] = walk_until_dis(center_left[0],
                                                          center_left[1],
                                                          slope_left,
                                                          dis_left,
                                                          'left' if slope_left > 0 else 'right',
                                                          img_size_yx=img_size_yx
                                                        )
            expanded_trapezoid_points[3] = walk_until_dis(center_left[0],
                                                          center_left[1],
                                                          slope_left,
                                                          dis_left,
                                                          'right' if slope_left > 0 else 'left',
                                                          img_size_yx=img_size_yx
                                                        )
        if i[1][0] - i[2][0] == 0:
            expanded_trapezoid_points[1] = center_right - dis_right * np.array([0,1])
            expanded_trapezoid_points[2] = center_right + dis_right * np.array([0,1])
            
        else:
            slope_right = (i[1][1] - i[2][1])/(i[1][0] - i[2][0])
            expanded_trapezoid_points[1] = walk_until_dis(center_right[0],
                                                          center_right[1],
                                                          slope_right,
                                                          dis_right,
                                                          'left' if slope_right > 0 else 'right',
                                                          img_size_yx=img_size_yx
                                                        )
            expanded_trapezoid_points[2] = walk_until_dis(center_right[0],
                                                          center_right[1],
                                                          slope_right,
                                                          dis_right,
                                                          'right' if slope_right > 0 else 'left',
                                                          img_size_yx=img_size_yx
                                                        )
            
        expanded_trapezoid_points = expanded_trapezoid_points.astype(np.int64)
        out_list.append(expanded_trapezoid_points)
        
    return out_list

class Plt_Dynamic_Window:
    def __init__(self):
        plt.ion()
        self.x = np.arange(0, 10, 0.1)
        self.y = np.sin(self.x)
    
    def update(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        plt.clf()
        plt.plot(self.x, self.y)
        plt.pause(0.001)
        plt.ioff()
        
        
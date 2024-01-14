#coding=utf-8
import numpy as np
import math
import cv2
import os
from typing import Optional
class Videosaver:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def byphone(cls,
                if_show:bool = True,
                if_save_img:bool = True,
                save_interval_frames:int = 10,
                save_img_folder:str = './img_capture_out',
                save_img_fmt:str = 'jpg',
                save_video_path:Optional[list] = None,
                codec :str = 'mp4v',
                path:str="rtsp://127.0.0.1:8080/h264_pcm.sdp"
                ):
        '''
        no img_processing
        '''
        
        count = 0
        
        vd=cv2.VideoCapture(path)
        if save_video_path is not None:
            fps = int(vd.get(cv2.CAP_PROP_FPS))
            width = int(vd.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vd.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_writer = cv2.VideoWriter()
            fourcc = video_writer.fourcc(*codec)
            video_writer.open(save_video_path,fourcc,fps,(width,height))
                
                
        while True:
            count +=1
            save_img_path = os.path.join(save_img_folder,f'{count}.{save_img_fmt}')
            ret, frame = vd.read()  
            if not ret:
                break
            if if_show:
                cv2.imshow('press esc to break', frame) 
            if save_video_path is not None:
                video_writer.write(frame)
                
            if count % save_interval_frames == 0 and if_save_img:
                print(save_img_path)
                cv2.imwrite(save_img_path,frame)
             
            if cv2.waitKey(1) & 0xFF ==27:
                break
        vd.release()  
        
        if save_video_path is not None:
            video_writer.release()
            
            
        cv2.destroyAllWindows() 
        
    def bymindvision(cls):
        pass
    
    
    
    
    
    
if __name__=="__main__":
    
    
    Videosaver.byphone()

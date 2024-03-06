import cv2
import numpy as np
import glob
from autoaim_alpha.autoaim_alpha.utils_network.data import *
from autoaim_alpha.autoaim_alpha.camera.mv_class import *



# Define a function to calibrate the camera and save the calibration result for later use
def calibrate_camera(img_path, square_size, pattern_size,save_path='camera_calibration.yaml'):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0)....,(6,5,0)
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Read in all the images
    images = glob.glob(img_path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dict_info = {'mtx': mtx, 'dist': dist}
    Data.save_dict_info_to_yaml(dict_info, save_path)

    return mtx, dist


if __name__ == '__main__':
    
    action = 'record'   # record or calibrate
    
    img_path = './tmp_calib_images/*.jpg'
    calibration_img_save_path = './tmp_calib_images'
    calibration_save_path = 'camera_calibration.yaml'
    camera_config_folder = './camera_config'
    
    if action == 'record':
        ca = Mindvision_Camera(output_format='bgr8',
                            camera_mode='Dbg',
                            camera_config_folder=None,
                            if_auto_exposure=True)
        
        
        ca.enable_save_img(calibration_img_save_path,
                        save_img_interval=None,
                        press_key_to_save='p')
        
        with Custome_Context('ca',ca):
            while True: 
                
                img = ca.get_img()
                cv2.imshow('img',img)
                
                key = cv2.waitKey(1)
                
                if key == ord('q'):
                    break
        
                
            
                
        cv2.destroyAllWindows()
    
    if action == 'calibrate':
        mtx, dist = calibrate_camera(img_path, square_size=(9,6), pattern_size=(9,6), save_path=calibration_save_path)
        print('mtx:\n', mtx)
        print('dist:\n', dist)
        
        
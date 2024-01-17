from autoaim_alpha.autoaim_alpha.img.detector import Armor_Detector
import cv2
if __name__ == '__main__':
    detector = Armor_Detector(
        armor_color='red',
        mode='Dbg',
        tradition_config_folder='./tmp_tradition_config',
        net_params_yaml='./tmp_net_params.yaml',
        save_roi_key='c'
    )
    
    img = cv2.imread('./armorred.png')
    result,t = detector.get_result(img,img)
    print(result)
    print(t)
    
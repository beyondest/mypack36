from autoaim_alpha.autoaim_alpha.img.detector import Armor_Detector
import cv2
import time
if __name__ == '__main__':
    detector = Armor_Detector(
        armor_color='red',
        mode='Dbg',
        tradition_config_folder='./autoaim_alpha/config/tradition_config',
        net_config_folder='./autoaim_alpha/config/net_config',
        depth_estimator_config_yaml='./autoaim_alpha/config/other_config/pnp_params.yaml',
    )
    img = cv2.imread('./armorred.png')
    img = cv2.resize(img,(640,640))
    result,t = detector.get_result(img)
    img = detector.visualize(img,round(1/t),None)
    cv2.imshow('result',img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(result)
    print(t)
    
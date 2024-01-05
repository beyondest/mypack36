import sys

sys.path.append('..')
import img.img_operation as imo
import numpy as np

import cv2


img = cv2.imread('./res/armorred.png')

center_list = imo.find_armor_beta(img)
imo.cvshow(img)

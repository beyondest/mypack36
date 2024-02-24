
# Notice : aspect = wid/hei, wid is the horizon side length of img, hei is vertical side length of img 


# FIlter of Lightbar
SMALL_LIGHTBAR_SINGLE_AREA_RANGE = [1,5*1000]
SMALL_LIGHTBAR_SINGLE_ASPECT_RATIO_RANGE = [0.01,1]
SMALL_LIGHTBAR_CENTER_DISTANCE_RANGE = [1,1000]
SMALL_LIGHTBAR_TWO_AREA_RATIO_RANGE = [0.23,5.61]
SMALL_LIGHTBAR_TWO_ASPECT_RATIO_RANGE = [0.28,3.09]


# Filter of Big Rec
BIG_REC_AREA_RANGE = [10,350*1000]
BIG_REC_ASPECT_RATIO_RANGE = [1,5]




# Preprocess
GAUSSIAN_BLUR_KERNAL_SIZE = [3,3]
GAUSSIAN_BLUR_X = 1
CLOSE_KERNEL_SIZE = [2,2]
STRECH_MAX = None

RED_ARMOR_YUV_RANGE = [152,255]
RED_ARMOR_BINARY_ROI_THRESHOLD = 5
BLUE_ARMOR_YUV_RANGE = [145,255]
BLUE_ARMOR_BINARY_ROI_THRESHOLD = 4


# Custom Operation
EXPAND_RATE = 1.5




# Net Params
NET_INPUT_SIZE = [32,32]
NET_INPUT_DTYPE = 'float32'
ENGINE_TYPE = 'ort'
NET_INPUT_NAME = 'inputs'
NET_OUTPUT_NAME = 'outputs'
NET_CONFIDENCE = 0.5

MAX_INPUT_BATCHSIZE = 10

# Depth Estimation

IMG_SHRINK_SCALE = 0.5
SMALL_ARMOR_REC_POINTS = [[-67., -27.5, 0.],
                            [67., -27.5, 0.],
                            [-67., 27.5, 0.],
                            [67., 27.5, 0.]]

BIG_ARMOR_REC_POINTS = [[-112.5, -27.5, 0.],
                        [112.5, -27.5, 0.],
                        [-112.5, 27.5, 0.],
                        [112.5, 27.5, 0.]]

MV1_MTX =  [[1.38012881e+03, 0.00000000e+00, 6.32563131e+02],
                [0.00000000e+00, 1.41745621e+03, 3.95972569e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


MV1_DIST = [[0.00417728, -0.30223262, -0.01761736, -0.00159902, 1.24798931]]

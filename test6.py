import math
import numpy as np
def change_abs_to_relative(abs_yaw, current_yaw):
    if abs(abs_yaw - current_yaw) < math.pi:
        return abs_yaw - current_yaw
    left_offset = 2 * math.pi - current_yaw + abs_yaw
    right_offset = 2 * math.pi - abs_yaw + current_yaw
    if left_offset < right_offset:
        return -left_offset
    else:
        return right_offset
    
    
for abs_yaw in np.arange(0, 2 * math.pi, 0.1):
    current_yaw = 0.5 * math.pi
    

    result = change_abs_to_relative(abs_yaw, current_yaw)
    print(f"abs_yaw: {abs_yaw:.3f}, current_yaw: {current_yaw:3f}, result: {result:.3f}")
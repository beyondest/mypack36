import cv2
import numpy as np


from autoaim_alpha.autoaim_alpha.img.tools import *
c = canvas((800,1600))



ret1 = np.array([[400,400],[450,400],[450,500],[400,500]])
ret2 = np.array([[600,600],[700,600],[700,750],[600,750]])

order_cont = get_trapezoid_corners(ret1,ret2)

print(order_cont)
order_cont = order_cont.astype(int)
expand_trapzezoid = expand_trapezoid_wid([order_cont],1.5,(800,1600))[0]
expand_trapzezoid = expand_trapzezoid.astype(int)

print(expand_trapzezoid)
draw_big_rec_list([ret1,ret2,expand_trapzezoid],c.img)
c.show()




import matplotlib
from matplotlib import pyplot as plt
import numpy as np
 
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
        
        
   
a = Plt_Dynamic_Window() 
# 创建循环
for i in range(30):
    x = np.arange(0, 10, 0.1) + i
    y = np.sin(x)
    a.update(x,y)
    

    
    
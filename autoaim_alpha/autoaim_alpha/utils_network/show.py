import numpy as np
import matplotlib.pyplot as plt
from utils_network.data import *
import time
def least_square_show():
    '''
    show linear regression figure
    '''
    def func0(x:np.ndarray)->np.ndarray:
        return np.power(x,2)
    
    def func1(x:np.ndarray)->np.ndarray:
        return np.exp(x)
    def func2(x:np.ndarray):
        return np.log(np.abs(x))
    def func3(x):
        return np.sin(x)
    def func4(x):
        return np.power(x,3)
    
    data_maker=Data(seed=int(time.perf_counter()))
    point_data=data_maker.random_point(2,1000,(-4,4),
                            [[[0],[0]],[[0],[1],func1]],
                            [[[1],"normal",(0,5)]]
                            )
    
    
    #Ax=y
    A=data_maker.make_taylor_basis(point_data.all_points[:,0],order=3)
    y=np.copy(point_data.all_points[:,1]).reshape(-1,1)

    t1=time.perf_counter()
    #A@pinv(A.T@A)@A.Ty=x_hex
    
    x_hex=np.linalg.pinv(A.T@A)@A.T@y
    y_hex=A@x_hex
    t2=time.perf_counter()
    
    print("timeis",t2-t1)
    
    #predict using best coefficient_array:x_hex;oder must be same
    axis_predict=np.linspace(4,6,1000).reshape(-1,1)
    B=data_maker.make_taylor_basis(axis_predict,order=3)
    y_predict=B@x_hex
    
    
    
    f=Data.plt_figure()
    ax_index=f.plt_point('plot',x=point_data.all_points[:,0],y=y_hex,color='r')
    ax_index=f.plt_point('scatter',ax_index=ax_index,x=point_data.all_points[:,0],y=point_data.all_points[:,1],color='b')
    ax_index=f.plt_point('plot',ax_index=ax_index,x=axis_predict,y=y_predict,color='g')
    plt.show()       


def gradient_show():
    '''power2d func show, z=x^2+y^2'''
    def gradient_power2d(x):

        gradient = np.array([2 * x[0], 2 * x[1]])
        return gradient
    def power2d(x):
        return np.power(x[0], 2) + np.power(x[1], 2)
      

    d=Data()
    point_data=d.random_point(3,100,(-3,3),[[[0,1],[2],power2d]])
    
    init_pt=np.array([2,2]).reshape(2,1)
    learning_rate=0.1
    iters=10
    
    current_x = init_pt
    plt.ion()
    f=Data.plt_figure()
    ax_index=f.plt_point(x=point_data.all_points[0],
                y=point_data.all_points[1],
                z=point_data.all_points[2],
                projection='3d',
                )
    for i in range(iters):
        gradient = gradient_power2d(current_x)

        current_x = current_x - learning_rate * gradient
        
        
        new_point=np.array([current_x[0],current_x[1],power2d(current_x)]).reshape(3,1)
        print(new_point[2])
        f.plt_point(x=new_point[0],y=new_point[1],z=new_point[2],
                    ax_index=ax_index,color='r',show_now=True)
        input('Press enter here in console to continue')
        #print(f"Iteration {i+1}: x = {current_x}, Power2D(x) = {power2d(current_x)}")
        
    return current_x



if __name__=='__main__':
    
    least_square_show()
    #gradient_show()
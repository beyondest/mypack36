import time

def timing(circle_times:int = 1, if_show_total:bool = False):
    """This is a timing decorator factory
    Return:list
        If circle_times is 1, then return[ result of ori_func, elapsed_time] 
        Elif circle_times bigger than 1, then return [last result of ori_func, average elapsed_time]
    """
    
    def decorator(func):
        def inner(*args, **kwargs):
            total_time = 0
            for i in range(circle_times):
                
                t1=time.perf_counter()
                result=func(*args, **kwargs)
                t2=time.perf_counter()
                total_time+=t2-t1
            avg_time = total_time/circle_times
            if if_show_total:
                print(f'total_spend_time in {circle_times} circles: {total_time}')
            return [result,avg_time]
        return inner
    return decorator


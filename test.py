from datetime import datetime
import time

current_time = datetime.now()

formatted_time = current_time.strftime("%M:%S")
print(formatted_time)
seconds_integer = int(formatted_time.split(":")[1])
seconds_fraction = time.time() % 1

time_info = [int(current_time.strftime("%M")), seconds_integer, round(seconds_fraction,4)]
print(time.perf_counter())
print(time_info)


print(time.time())



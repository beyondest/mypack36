from autoaim_alpha.autoaim_alpha.decision_maker.ballistic_predictor import *

import matplotlib.pyplot as plt
"""
Calculate air drag through bullet drop on ground time with init_pitch 

pitch = pi/4

k    : t
1e-4 : 2.94
1e-5 : 3.75
1e-6 : 3.90


Z                    : 11
bullet_flight_time   : 0.48
pitch                : 0.071        4deg

Z                    : 0.5
bullet_flight_time   : 0.02
pitch                : -0.29         -16deg
"""


b = Ballistic_Predictor(params_yaml_path="ballistic_params.yaml",
                        if_show_ballistic_trajectory=False)

#b.cal_max_shooting_hei_above_ground(b.params.pitch_range[1])
#b.cal_max_shooting_dis_by_gradient()
#b.save_params_to_yaml("ballistic_params.yaml")
x = []
acc_hei_diff_list = []
no_acc_hei_diff_list = []
acc_flight_time_list = []
no_acc_flight_time_list = []
for i in np.arange(0,1e-4,1e-7):
    
    b.params.k = i
    
    target_pos = np.array([0, -0.3, 5])
    cur_pitch = 0
    cur_yaw = 0
    b._update_camera_pos_in_gun_pivot_frame(cur_yaw, cur_pitch)
    target_pos_in_gun_pivot_frame = b.params.camera_pos_in_gun_pivot_frame + target_pos
    tvec_yoz = target_pos_in_gun_pivot_frame[[1,2]]
    
    
    
        
    b.params.R_K4_dt_near = 0.0001
    b.params.R_K4_dt_far = 0.0001
    b.params.R_K4_error_tolerance = 0.003

    # most accurate result
    accurate_result = b.get_fire_yaw_pitch(target_pos,cur_yaw,cur_pitch)

    [acc_hei_diff,acc_flight_time],_ = b._R_K4_air_drag_ballistic_model(0,tvec_yoz,'hei')
    print(f"Accurate Hei Diff: {acc_hei_diff},flight time: {acc_flight_time}")

    x.append(i)
    acc_hei_diff_list.append(acc_hei_diff)
    acc_flight_time_list.append(acc_flight_time)
    
plt.figure()
plt.plot(x,acc_hei_diff_list,'r',label="Accurate")
plt.xlabel("k")
plt.ylabel("Hei Diff")
plt.legend()



plt.figure()
plt.plot(x,acc_flight_time_list,'r',label="Accurate")
plt.xlabel("k")
plt.ylabel("Flight Time")
plt.legend()

plt.show()


    







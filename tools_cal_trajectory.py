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


b = Ballistic_Predictor(params_yaml_path="autoaim_alpha/config/other_config/ballistic_params.yaml",
                        if_show_ballistic_trajectory=False)

#b.cal_max_shooting_hei_above_ground(b.params.pitch_range[1])
#b.cal_max_shooting_dis_by_gradient()
#b.save_params_to_yaml("ballistic_params.yaml")
x = []
solve_time = []
hei_error = []
fllight_time = []
fllight_time2 = []
solve_result = []
for i in np.arange(0.25,11,0.01):
    target_pos = np.array([0, -0.3, i])

    cur_pitch = 0
    cur_yaw = 0

    b._update_camera_pos_in_gun_pivot_frame(cur_yaw, cur_pitch)
    target_pos_in_gun_pivot_frame = b.params.camera_pos_in_gun_pivot_frame + target_pos
    tvec_yoz = target_pos_in_gun_pivot_frame[[1,2]]
    target_z_in_gun_pivot_frame = tvec_yoz[0]
    print(f"Target Z: {target_z_in_gun_pivot_frame}")
    
    # not accurate but fast result
    t1 = time.perf_counter()
    fast_result = b.get_fire_yaw_pitch(target_pos,cur_yaw,cur_pitch)
    t2 = time.perf_counter()
    

    b.params.R_K4_dt_near = 0.0001
    b.params.R_K4_dt_far = 0.0001
    b.params.R_K4_error_tolerance = 0.003

    # most accurate result
    #accurate_result = b.get_fire_yaw_pitch(target_pos,cur_yaw,cur_pitch)

    [hei_diff,t],_ = b._R_K4_air_drag_ballistic_model(fast_result[1],tvec_yoz)
    print(hei_diff,t)

    #[hei_diff,t],solve_time = b._R_K4_air_drag_ballistic_model(accurate_result[1],tvec_yoz)
    #print(hei_diff,t,solve_time)
    x.append(target_z_in_gun_pivot_frame)
    hei_error.append(hei_diff)
    fllight_time.append(t)
    fllight_time2.append(fast_result[2])
    solve_time.append(t2-t1)
    solve_result.append(fast_result[3])
    b.params.R_K4_dt_near = 0.001
    b.params.R_K4_dt_far = 0.005
    b.params.R_K4_error_tolerance = 0.03




plt.figure()
plt.plot(x,hei_error)
plt.xlabel("Target Z (m)")
plt.ylabel("Height Error (m)")
plt.title("Height Error vs Target Z")

plt.figure()
plt.plot(x,fllight_time,'r-',label = 'accurate flight time')
plt.plot(x,fllight_time2,'b--',label = 'not accurate flight time')
plt.xlabel("Target Z (m)")
plt.ylabel("Flight Time (s)")
plt.title("Flight Time vs Target Z")

plt.figure()
plt.plot(x,solve_time)
plt.xlabel("Target Z (m)")
plt.ylabel("Solve Time (s)")
plt.title("Solve Time vs Target Z")


plt.figure()
plt.plot(x,solve_result)
plt.xlabel("Target Z (m)")
plt.ylabel("Solve Result")
plt.title("Solve Result vs Target Z")

plt.show()  






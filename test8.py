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
tra = []

target_pos = np.array([0, -0.3, 0.4])

cur_pitch = 0
cur_yaw = 0

b._update_camera_pos_in_gun_pivot_frame(cur_yaw, cur_pitch)
target_pos_in_gun_pivot_frame = b.params.camera_pos_in_gun_pivot_frame + target_pos
tvec_zoy = target_pos_in_gun_pivot_frame[[2,1]]
target_z_in_gun_pivot_frame = tvec_zoy[0]
print(f"Target Z: {target_z_in_gun_pivot_frame}")

# not accurate but fast result
fast_result = b.get_fire_yaw_pitch(target_pos,cur_yaw,cur_pitch)

b.params.R_K4_dt_near = 0.0001
b.params.R_K4_error_tolerance = 0.005

# most accurate result
#accurate_result = b.get_fire_yaw_pitch(target_pos,cur_yaw,cur_pitch)

[hei_diff,t],solve_time = b._R_K4_air_drag_ballistic_model(fast_result[1],tvec_zoy)
print(hei_diff,t,solve_time)

#[hei_diff,t],solve_time = b._R_K4_air_drag_ballistic_model(accurate_result[1],tvec_zoy)
#print(hei_diff,t,solve_time)


pt = np.array([target_z_in_gun_pivot_frame,hei_diff])
b.params.R_K4_dt = 0.001
b.params.R_K4_error_tolerance = 0.1









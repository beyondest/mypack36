from autoaim_alpha.autoaim_alpha.img.depth_estimator import *

a = Depth_Estimator(None)
a.pnp_params.print_show_all()

print(np.array(a.pnp_params.mtx))
print(np.array(a.pnp_params.dist))



import numpy as np

from plane_ode import PlaneODE2D, n_traj
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dymos import Phase
from itertools import combinations

np.random.seed(12345)

p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
#p.driver.opt_settings['Major iterations limit'] = 10000
#p.driver.opt_settings['Minor iterations limit'] = 10000
#p.driver.opt_settings['Iterations limit'] = 100000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
#p.driver.opt_settings["Linesearch tolerance"] = 0.01
#p.driver.opt_settings["Major step limit"] = 0.1
p.driver.opt_settings['iSumm'] = 6
'gauss-lobatto'

phase = Phase(transcription='radau-ps',
              ode_class=PlaneODE2D,
              num_segments=20,
              transcription_order=3,
              compressed=True)

p.model.add_subsystem('phase0', phase)
p.model.options['assembled_jac_type'] = 'csc'
#p.model.linear_solver = DirectSolver(assemble_jac=True)

phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(1, 1500))


locations = []
for i in range(n_traj):
    # trajectories random start/end locations in circle of radius 4000 around this point
    center_x = 0
    center_y = 0
    theta = np.random.uniform(0, 2*np.pi)
    theta2 = np.random.uniform(np.pi-np.pi/5, np.pi + np.pi/5)
    r = 4000.
    start_x = center_x + r*np.cos(theta)
    start_y = center_y + r*np.sin(theta)

    end_x = center_x + r*np.cos(theta + theta2)
    end_y = center_y + r*np.sin(theta + theta2)

    locations.append([start_x, end_x, start_y, end_y])
    phase.set_state_options('x%d' % i, lower=-10000, upper=10000,
                            scaler=0.001, defect_scaler=.0001)
    phase.set_state_options('y%d' % i, lower=-10000, upper=10000,
                            scaler=0.001, defect_scaler=.0001)

    phase.add_boundary_constraint('x%d' % i, loc='initial', equals=start_x)
    phase.add_boundary_constraint('y%d' % i, loc='initial', equals=start_y)
    phase.add_boundary_constraint('x%d' % i, loc='final', equals=end_x)
    phase.add_boundary_constraint('y%d' % i, loc='final', equals=end_y)
    # phase.set_state_options('v%d' % i, fix_initial=False, fix_final=False,
    #                     scaler=0.01, defect_scaler=0.01, lower=0.0)

    phase.add_control('v%d' % i, units='m/s', opt=False, upper=10, lower=0.0)
    phase.add_control('chi%d' % i, units='deg', opt=True, upper=180, lower=-180)

for i, j in combinations([i for i in range(n_traj)], 2):
    phase.add_path_constraint(name='distance_%d_%d.dist' % (i, j), 
                              constraint_name='distance_%d_%d' % (i, j), 
                              lower=6000.0, 
                              scaler=0.01,
                              units='m')

# Minimize time to target
phase.add_objective('time', loc='final', scaler=1.0)


p.setup()

phase = p.model.phase0

# range = 1019 at v0 = 100.0
#p['phase0.target_x'] = phase.interpolate(ys=[0, 700.0], nodes='parameter_input')
# p['phase0.target_y'] = 350.
# p['phase0.target_h'] = 50.


p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = 500.0

for i in range(n_traj):

    # trajectories random start/end locations in circle of radius 4000 around this point
    start_x, end_x, start_y, end_y = locations[i]

    p['phase0.states:x%d' % i] = phase.interpolate(ys=[start_x, end_x], nodes='state_input')
    p['phase0.states:y%d' % i] = phase.interpolate(ys=[start_y, end_y], nodes='state_input')
    #p['phase0.states:v%d' % i] = phase.interpolate(ys=[1.0, 1.0], nodes='state_input')


len_x = (p['phase0.states:x%d' % i][-1] - p['phase0.states:x%d' % i][0])[0]
len_y = (p['phase0.states:y%d' % i][-1] - p['phase0.states:y%d' % i][0])[0]

p.run_driver()

# ‘vode’, ‘lsoda’, ‘dopri5’
exp_out = phase.simulate(times='all', integrator='vode', record_file='sim.db')

data = {}
for i in range(n_traj):
    data[i] = {}
    data[i]['x'] = exp_out.get_values('x%d' % i, units='m').flatten()
    data[i]['y'] = exp_out.get_values('y%d' % i, units='m').flatten()

    data[i]['x_imp'] = phase.get_values('x%d' % i, units='m', nodes='all').flatten()
    data[i]['y_imp'] = phase.get_values('y%d' % i, units='m', nodes='all').flatten()

    #data[i]['T'] = exp_out.get_values('T%d' % i, units='N').flatten()
    data[i]['chi'] = exp_out.get_values('chi%d' % i, units='deg').flatten()
    data[i]['v'] = exp_out.get_values('v%d' % i, units='m/s').flatten()

    #data[i]['T_imp'] = phase.get_values('T%d' % i, units='N', nodes='all').flatten()
    data[i]['chi_imp'] = phase.get_values('chi%d' % i, units='deg', nodes='all').flatten()
    data[i]['v_imp'] = phase.get_values('v%d' % i, units='m/s', nodes='all').flatten()
    print("v%d" % i, min(data[i]['v']), max(data[i]['v']))
    print("chi%d" % i, min(data[i]['chi']), max(data[i]['chi']))


data['t'] = exp_out.get_values('time', units='s').flatten()
data['t_imp'] = phase.get_values('time', units='s').flatten()
print("time", data['t'][-1])
for i, j in combinations([i for i in range(n_traj)], 2):
    dist = exp_out.get_values('distance_%d_%d.dist' % (i, j), units='m').flatten()
    print(i, j, min(dist))

fig = plt.figure()


for i in range(n_traj):
    plt.plot(data[i]['x_imp'], data[i]['y_imp'], 'gray')
    plt.scatter(data[i]['x_imp'], data[i]['y_imp'], cmap='Greens', c=data['t_imp'])

plt.xlabel('x')
plt.ylabel('y')

plt.show()




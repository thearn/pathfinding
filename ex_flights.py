import numpy as np

from plane_ode import PlaneODE2D, n_traj, x_loc, y_loc, keepout_radius, personal_zone

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dymos import Phase
from itertools import combinations

import pickle

np.random.seed(4312)

p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
p.driver.opt_settings['Major iterations limit'] = 10000
p.driver.opt_settings['Minor iterations limit'] = 10000
p.driver.opt_settings['Iterations limit'] = 100000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-10
#p.driver.opt_settings["Linesearch tolerance"] = 0.01
p.driver.opt_settings["Major step limit"] = 0.1
p.driver.opt_settings['iSumm'] = 6




phase = Phase(transcription='gauss-lobatto',
              ode_class=PlaneODE2D,
              num_segments=35,
              transcription_order=3,
              compressed=True)

p.model.add_subsystem('phase0', phase)
#p.model.options['assembled_jac_type'] = 'csc'
#p.model.linear_solver = DirectSolver(assemble_jac=True)

max_time = 6500.0
phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(1, max_time))


locations = []
thetas = []

thetas = np.linspace(0, 2*np.pi, n_traj + 1)

schedules = []

for i in range(n_traj):
    # trajectories random start/end locations in circle of radius 4000 around this point
    center_x = 0
    center_y = 0
    # while True:
    #     theta = np.random.uniform(0, 1.5*np.pi)
    #     if len(thetas) == 0:
    #         thetas.append(theta)
    #         break
    #     elif min([abs(theta - t) for t in thetas]) > 0.6:
    #         thetas.append(theta)
    #         break

    t_start = np.random.uniform(1, max_time/4.0)
    t_end = np.random.uniform(1.2*max_time/2.0, max_time)
    schedules.append([t_start, t_end])
    print("schedule:", t_start, t_end)

    theta = thetas[i]        
    theta2 = np.random.uniform(np.pi-np.pi/6, np.pi + np.pi/6)
    r = 4000.

    if i%3 == 0: 
        theta2 = theta - np.pi / 4


    start_x = center_x + r*np.cos(theta)
    start_y = center_y + r*np.sin(theta)

    end_x = center_x + r*np.cos(theta + theta2)
    end_y = center_y + r*np.sin(theta + theta2)

    locations.append([start_x, end_x, start_y, end_y])
    #print("location", i, locations[-1])
    phase.add_input_parameter('sx%d'%i, val=start_x, units='m')
    phase.add_input_parameter('sy%d'%i, val=start_y, units='m')
    phase.add_input_parameter('ex%d'%i, val=end_x, units='m')
    phase.add_input_parameter('ey%d'%i, val=end_y, units='m')

    phase.add_input_parameter('ts%d'%i, val=t_start, units='s')
    phase.add_input_parameter('te%d'%i, val=t_end, units='s')


    phase.set_state_options('x%d' % i,
                            scaler=0.01, defect_scaler=0.1)
    phase.set_state_options('y%d' % i,
                            scaler=0.01, defect_scaler=0.1)

    phase.add_boundary_constraint('x%d' % i, loc='initial', equals=start_x)
    phase.add_boundary_constraint('y%d' % i, loc='initial', equals=start_y)
    phase.add_boundary_constraint('x%d' % i, loc='final', equals=end_x)
    phase.add_boundary_constraint('y%d' % i, loc='final', equals=end_y)
    # phase.set_state_options('v%d' % i, fix_initial=False, fix_final=False,
    #                     scaler=0.01, defect_scaler=0.01, lower=0.0)

    phase.add_control('vx%d' % i, rate_continuity=False, units='m/s', 
                      opt=True, upper=10, lower=-10.0, scaler=200.0, adder=-10)
    phase.add_control('vy%d' % i, rate_continuity=False, units='m/s', 
                      opt=True, upper=10, lower=-10.0, scaler=200.0, adder=-10)
    #phase.add_control('chi%d' % i, units='deg', opt=True, upper=180, lower=-180)
    phase.add_path_constraint(name = 'schedule%d.err_d' % i,
                              constraint_name = 'schedule_err%d' % i,
                              scaler=100.0,
                              upper=10.0,
                              units='m')
    phase.add_path_constraint(name = 'keepout%d.dist' % i,
                              constraint_name = 'keepout%d' % i,
                              scaler=0.1,
                              lower=keepout_radius,
                              units='m')
for i, j in combinations([i for i in range(n_traj)], 2):
    phase.add_path_constraint(name='distance_%d_%d.dist' % (i, j), 
                              constraint_name='distance_%d_%d' % (i, j), 
                              scaler=0.1,
                              lower=personal_zone, 
                              units='m')

# Minimize time to target
phase.add_objective('time', loc='final', scaler=0.1)
#phase.add_objective('vtotals.vtotal', loc='final', scaler=1.0) #71000
#phase.add_objective('schedule0.err_d', loc='final', scaler=10.0)

p.setup()

phase = p.model.phase0

# range = 1019 at v0 = 100.0
#p['phase0.target_x'] = phase.interpolate(ys=[0, 700.0], nodes='parameter_input')
# p['phase0.target_y'] = 350.
# p['phase0.target_h'] = 50.


p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = max_time

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
exp_out = phase.simulate(times='all', record=False)

data = {}
data['t'] = exp_out.get_val('phase0.timeseries.time', units='s').flatten()
for i in range(n_traj):
    data[i] = {}
    data[i]['x'] = exp_out.get_val('phase0.timeseries.states:x%d' % i, units='m').flatten()
    data[i]['y'] = exp_out.get_val('phase0.timeseries.states:y%d' % i, units='m').flatten()
    data[i]['loc'] = locations[i]

with open('sim.pkl', 'wb') as f:
    pickle.dump(data, f)

# print("time", data['t'][-1])
# for i, j in combinations([i for i in range(n_traj)], 2):
#     dist = exp_out.get_val('phase0.timeseries.distance_%d_%d.dist' % (i, j), units='m').flatten()
#     print(i, j, min(dist))

circle_x = []
circle_y = []
r = 4000.
for i in np.linspace(0, 2*np.pi, 1000):
    circle_x.append(r*np.cos(i))
    circle_y.append(r*np.sin(i))


kcircle_x = []
kcircle_y = []
r = keepout_radius
for i in np.linspace(0, 2*np.pi, 1000):
    kcircle_x.append(x_loc + r*np.cos(i))
    kcircle_y.append(y_loc + r*np.sin(i))


fig = plt.figure()

plt.plot(kcircle_x, kcircle_y, 'k--', linewidth=0.5)
plt.plot(circle_x, circle_y, 'k-.', linewidth=0.5)


for sx, ex, sy, ey in locations:
    plt.plot([sx, ex], [sy, ey], 'kx', markersize=10)
for i in range(n_traj):
    plt.plot(data[i]['x'], data[i]['y'], 'gray')
    plt.scatter(data[i]['x'], data[i]['y'], cmap='Greens', c=data['t'])

plt.xlabel('x')
plt.ylabel('y')

plt.show()




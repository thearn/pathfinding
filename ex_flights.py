import numpy as np

from plane_ode import PlaneODE2D, n_traj, x_loc, y_loc, keepout_radius, personal_zone

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from dymos import Phase
from itertools import combinations

import pickle

np.random.seed(432)

p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
#p.driver.set_simul_deriv_color('coloring.json')
#p.driver.opt_settings['Start'] = 'Cold'
p.driver.opt_settings['iSumm'] = 6

p.driver.opt_settings['Iterations limit'] = 100000
p.driver.opt_settings["Major step limit"] = 0.5 #2.0
p.driver.opt_settings['Major iterations limit'] = 10000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-4

p.driver.opt_settings['Minor iterations limit'] = 10000
p.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4

p.driver.opt_settings['Scale option'] = 0 #0

p.driver.opt_settings['QPSolver'] = 'QN' #Cholesky, QN, CG
p.driver.opt_settings['Partial price'] = 10 # 1 to 10

p.driver.opt_settings['Crash option'] = 3 # 3 1 2
#p.driver.opt_settings["Linesearch tolerance"] = 0.1

#p.driver.opt_settings['New superbasics limit'] = 99 # 99
p.driver.opt_settings['Proximal point method'] = 1 # 1 2
p.driver.opt_settings['Violation limit'] = 10.0 # 10.0

# p.driver.opt_settings['']
# p.driver.opt_settings['']
# p.driver.opt_settings['']




phase = Phase(transcription='gauss-lobatto',
              ode_class=PlaneODE2D,
              num_segments=20,
              transcription_order=3,
              compressed=True)

p.model.add_subsystem('phase0', phase)
#p.model.options['assembled_jac_type'] = 'csc'
#p.model.linear_solver = DirectSolver(assemble_jac=True)

max_time = 6500.0
phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(max_time/4, max_time))


locations = []
thetas = []

rand_st = np.random.uniform(0, 2*np.pi)
thetas = np.linspace(rand_st, rand_st + 2*np.pi, 2*n_traj + 1)[:-1]
np.random.shuffle(thetas)

schedules = []
r = 4000.0
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

    theta = thetas[2*i]
    theta2 = thetas[2*i + 1]


    start_x = center_x + r*np.cos(theta)
    start_y = center_y + r*np.sin(theta)

    end_x = center_x + r*np.cos(theta2)
    end_y = center_y + r*np.sin(theta2)

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
phase.add_path_constraint(name = 'keepout.err_keepoutdist',
                          constraint_name = 'keepoutdist',
                          scaler=10.0,
                          upper=0.01)

phase.add_path_constraint(name='mdist.err_dist', 
                          constraint_name='mdist_err_dist', 
                          scaler=10.0,
                          upper=0.01)

# Minimize time to target
#phase.add_objective('time', loc='final', scaler=0.1)
phase.add_objective('summary.total_dist', loc='final', scaler=0.001) #71000

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
print("time:", data['t'][-1])
for i in range(n_traj):
    data[i] = {}
    data[i]['x'] = exp_out.get_val('phase0.timeseries.states:x%d' % i, units='m').flatten()
    data[i]['y'] = exp_out.get_val('phase0.timeseries.states:y%d' % i, units='m').flatten()
    data[i]['loc'] = locations[i]

with open('sim.pkl', 'wb') as f:
    pickle.dump(data, f)


fig = plt.figure()
ax = plt.gca()

circle = plt.Circle((0, 0), 4000, fill=False)
ax.add_artist(circle)

circle = plt.Circle((x_loc, y_loc), keepout_radius, fill=False, hatch='xxx')
ax.add_artist(circle)

for sx, ex, sy, ey in locations:
    plt.plot([sx, ex], [sy, ey], 'kx', markersize=10)
for i in range(n_traj):
    plt.plot(data[i]['x'], data[i]['y'], 'gray')
    plt.scatter(data[i]['x'], data[i]['y'], cmap='Greens', c=data['t'])

plt.tight_layout(pad=1)
plt.axis('equal')
plt.xlim(-4100,4100)
plt.ylim(-4100,4100)

plt.xlabel('x')
plt.ylabel('y')

plt.show()




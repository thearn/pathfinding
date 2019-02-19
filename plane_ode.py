import numpy as np

from openmdao.api import ExplicitComponent, Group
from flight_path_eom_2d import FlightPathEOM2D
from dymos import Phase, ODEOptions
from distance import Distance, VSum, KeepOut
from schedule import Schedule
from dymos import declare_time, declare_state, declare_parameter
from itertools import combinations

n_traj = 8
x_loc = 0.0
y_loc = 0.0
keepout_radius = 1500.0
ks_start = 600.0
personal_zone = 500.0

class PlaneODE2D(Group):
    ode_options = ODEOptions()

    ode_options.declare_time(units='s', targets = ['keepout%d.time' % i for i in range(n_traj)] + ['schedule%d.time' % i for i in range(n_traj)])

    targets = {}
    for i in range(n_traj):
        targets[i] = {'x': ['keepout%d.x' % i, 'schedule%d.x' % i], 'y' : ['keepout%d.y' % i, 'schedule%d.y' % i]}

    for i, j in combinations([i for i in range(n_traj)], 2):
        targets[i]['x'].append('distance_%d_%d.x1' % (i, j))
        targets[i]['y'].append('distance_%d_%d.y1' % (i, j))

        targets[j]['x'].append('distance_%d_%d.x2' % (i, j))
        targets[j]['y'].append('distance_%d_%d.y2' % (i, j))

    # dynamic trajectories
    for i in range(n_traj):
        ode_options.declare_state(name='x%d' % i, rate_source='flight_path%d.x_dot' % i, targets=targets[i]['x'], units='m')
        ode_options.declare_state(name='y%d' % i, rate_source='flight_path%d.y_dot' % i, targets=targets[i]['y'], units='m')
        #ode_options.declare_state(name='L%d' % i, rate_source='flight_path%d.L_dot' % i, units='m')
        ode_options.declare_parameter(name='vx%d' % i, targets = 'flight_path%d.vx' % i, units='m/s')
        ode_options.declare_parameter(name='vy%d' % i, targets = 'flight_path%d.vy' % i, units='m/s')

        ode_options.declare_parameter(name='sx%d' % i, targets = 'schedule%d.x_start' % i, units='m')
        ode_options.declare_parameter(name='sy%d' % i, targets = 'schedule%d.y_start' % i, units='m')
        ode_options.declare_parameter(name='ex%d' % i, targets = 'schedule%d.x_end' % i, units='m')
        ode_options.declare_parameter(name='ey%d' % i, targets = 'schedule%d.y_end' % i, units='m')

        ode_options.declare_parameter(name='ts%d' % i, targets = 'schedule%d.t_departure' % i, units='s')
        ode_options.declare_parameter(name='te%d' % i, targets = 'schedule%d.t_arrival' % i, units='s')

    def initialize(self):   
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('vtotals', subsys=VSum(n_traj=n_traj))

        for i in range(n_traj):
            self.add_subsystem(name='flight_path%d' % i,
                           subsys=FlightPathEOM2D(num_nodes=nn))
            self.connect('flight_path%d.vt' % i, 'vtotals.v%d' % i)
            
            self.add_subsystem(name='schedule%d' % i,
                           subsys=Schedule(num_nodes=nn))

            self.add_subsystem('keepout%d' % i, subsys=KeepOut(num_nodes=nn, x_loc=x_loc, y_loc=y_loc, ts = ks_start))
            

        traj = [i for i in range(n_traj)]
        for i, j in combinations(traj, 2):
            self.add_subsystem('distance_%d_%d' % (i,j), subsys=Distance(num_nodes=nn))
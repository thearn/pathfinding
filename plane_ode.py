import numpy as np

from openmdao.api import ExplicitComponent, Group
from flight_path_eom_2d import FlightPathEOM2D
from dymos import Phase, ODEOptions
from distance import Distance
from dymos import declare_time, declare_state, declare_parameter
from itertools import combinations

n_traj = 2




class PlaneODE2D(Group):
    ode_options = ODEOptions()

    ode_options.declare_time(units='s')

    targets = {}
    for i in range(n_traj):
        targets[i] = {'x': [], 'y' : []}

    for i, j in combinations([i for i in range(n_traj)], 2):
        targets[i]['x'].append('distance_%d_%d.x1' % (i, j))
        targets[i]['y'].append('distance_%d_%d.y1' % (i, j))

        targets[j]['x'].append('distance_%d_%d.x2' % (i, j))
        targets[j]['y'].append('distance_%d_%d.y2' % (i, j))

    # dynamic trajectories
    for i in range(n_traj):
        ode_options.declare_state(name='x%d' % i, rate_source='flight_path%d.x_dot' % i, targets=targets[i]['x'], units='m')
        ode_options.declare_state(name='y%d' % i, rate_source='flight_path%d.y_dot' % i, targets=targets[i]['y'], units='m')
        #ode_options.declare_state(name='v%d' % i, rate_source='flight_path%d.v_dot' % i, targets='flight_path%d.v' % i, units='m/s')
        ode_options.declare_parameter(name='chi%d' % i, targets = 'flight_path%d.chi' % i, units='rad')
        #ode_options.declare_parameter(name='T%d' % i, targets = 'flight_path%d.T' % i, units='N')
        ode_options.declare_parameter(name='v%d' % i, targets = 'flight_path%d.v' % i, units='m/s')

    def initialize(self):   
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        for i in range(n_traj):
            self.add_subsystem(name='flight_path%d' % i,
                           subsys=FlightPathEOM2D(num_nodes=nn))

        traj = [i for i in range(n_traj)]
        for i, j in combinations(traj, 2):
            self.add_subsystem('distance_%d_%d' % (i,j), subsys=Distance(num_nodes=nn))
import numpy as np

from openmdao.api import ExplicitComponent, Group
from flight_path_eom_2d import FlightPathEOM2D
from dymos import Phase, ODEOptions
from distance import KeepOut
from summary import ScheduleSum
from mult_distance import MDist
from schedule import Schedule
from dymos import declare_time, declare_state, declare_parameter
from itertools import combinations

n_traj = 4
x_loc = 1000.0
y_loc = -200.0
keepout_radius = 1500.0
ks_start = 900.0
personal_zone = 500.0

class PlaneODE2D(Group):
    ode_options = ODEOptions()

    ode_options.declare_time(units='s', targets = ['keepout.time'] + ['schedule%d.time' % i for i in range(n_traj)])

    targets = {}
    for i in range(n_traj):
        targets[i] = {'x': ['keepout.x%d' % i, 
                            'schedule%d.x' % i,
                            'mdist.x%d' % i], 
                      'y' : ['keepout.y%d' % i, 
                             'schedule%d.y' % i,
                             'mdist.y%d' % i]}

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

        for i in range(n_traj):
            self.add_subsystem(name='flight_path%d' % i,
                           subsys=FlightPathEOM2D(num_nodes=nn))
            
            self.add_subsystem(name='schedule%d' % i,
                           subsys=Schedule(num_nodes=nn))
            self.connect('schedule%d.destination_dist' % i, 'summary.destination_dist%d' % i)

        self.add_subsystem('summary', subsys=ScheduleSum(n_traj=n_traj))
        self.add_subsystem('keepout', subsys=KeepOut(num_nodes=nn, 
                                                     n_traj=n_traj, 
                                                     x_loc=x_loc, 
                                                     y_loc=y_loc,
                                                     keepout_radius=keepout_radius,
                                                     ts = ks_start))
        
        self.add_subsystem('mdist', subsys=MDist(n_traj=n_traj, num_nodes=nn))


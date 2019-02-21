import numpy as np

from openmdao.api import ExplicitComponent
from numpy import tan

mn = 0.1

class KeepOut(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('x_loc', types=float)
        self.options.declare('y_loc', types=float)
        self.options.declare('ts', types=float)


    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='x',
                       val=np.zeros(nn),
                       units='m')
        self.add_input(name='y',
                       val=np.zeros(nn),
                       units='m')

        self.add_input(name='time',
                       val=np.zeros(nn),
                       units='s')

        self.add_output(name='dist',
                       val=np.zeros(nn),
                       units='m')        

        ar = np.arange(nn)

        self.declare_partials('dist', 'x', rows=ar, cols=ar)
        self.declare_partials('dist', 'y', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        t = inputs['time']
        xl = self.options['x_loc']
        yl = self.options['y_loc']
        ts = self.options['ts']

        outputs['dist'] = np.sqrt((x - xl)**2 + (y - yl)**2)

        outputs['dist'][np.where(t <= ts)] = 4000.0

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        xl = self.options['x_loc']
        yl = self.options['y_loc']

        dist = np.sqrt((x - xl)**2 + (y - yl)**2)
        dist[np.where(dist < mn)] = mn

        partials['dist', 'x'] = (x - xl)/dist
        partials['dist', 'y'] = (y - yl)/dist




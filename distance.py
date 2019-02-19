import numpy as np

from openmdao.api import ExplicitComponent
from numpy import tan

mn = 0.1

class VSum(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_traj', types=int)

    def setup(self):
        n_traj = self.options['n_traj']
        self.vnames = []
        for i in range(n_traj):
            self.add_input(name='v%d' % i, val=0.0)
            self.vnames.append('v%d' % i)
        self.add_output(name='vtotal', val=0.0)

        self.declare_partials('vtotal', self.vnames)

    def compute(self, inputs, outputs):
        for name in self.vnames:
            outputs['vtotal'] += inputs[name]


    def compute_partials(self, inputs, partials):
        for name in self.vnames:
            partials['vtotal', name] = 1.0

class Distance(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='x1',
                       val=np.zeros(nn),
                       units='m')
        self.add_input(name='x2',
                       val=2*np.ones(nn),
                       units='m')

        self.add_input(name='y1',
                       val=3*np.ones(nn),
                       units='m')
        self.add_input(name='y2',
                       val=4*np.ones(nn),
                       units='m')

        self.add_output(name='dist',
                       val=np.zeros(nn),
                       units='m')        

        ar = np.arange(nn)

        self.declare_partials('dist', 'x1', rows=ar, cols=ar)
        self.declare_partials('dist', 'y1', rows=ar, cols=ar)
        self.declare_partials('dist', 'x2', rows=ar, cols=ar)
        self.declare_partials('dist', 'y2', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = inputs['y1']
        y2 = inputs['y2']

        outputs['dist'] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    def compute_partials(self, inputs, partials):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = inputs['y1']
        y2 = inputs['y2']

        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        dist[np.where(dist < mn)] = mn

        partials['dist', 'x1'] = (x1 - x2)/dist
        partials['dist', 'x2'] = (-x1 + x2)/dist
        partials['dist', 'y1'] = (y1 - y2)/dist
        partials['dist', 'y2'] = (-y1 + y2)/dist



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




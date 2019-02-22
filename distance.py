import numpy as np

from openmdao.api import ExplicitComponent
from mult_distance import loss

import matplotlib.pyplot as plt

def activate(t, ts, a=1.0):
    y = (np.tanh((t - ts)*a) + 1) / 2.0
    #a = a / 100
    dy = 0.5*a*(-np.tanh(a*(t - ts))**2 + 1)
    return y, dy


# t = np.linspace(1,100,100)

# y,dy = activate(t, 30)

# plt.plot(t, y)
# plt.plot(t, dy)
# plt.show()
# quit()

class KeepOut(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_traj', types=int)
        self.options.declare('num_nodes', types=int)
        self.options.declare('x_loc', types=float)
        self.options.declare('y_loc', types=float)
        self.options.declare('ts', types=float)
        self.options.declare('keepout_radius', types=float)


    def setup(self):
        nn = self.options['num_nodes']
        n_traj = self.options['n_traj']

        self.vnames = []
        for i in range(n_traj):
            self.add_input(name='x%d' % i, val=np.zeros(nn), units='m')
            self.add_input(name='y%d' % i, val=np.zeros(nn), units='m')
            self.vnames.extend(['x%d' % i, 'y%d' % i])

        self.add_input(name='time',
                       val=np.zeros(nn),
                       units='s')

        self.add_output(name='err_keepoutdist',
                       val=np.zeros(nn))        

        ar = np.arange(nn)

        self.declare_partials('err_keepoutdist', self.vnames, rows=ar, cols=ar)
        self.declare_partials('err_keepoutdist', 'time', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        t = inputs['time']
        xl = self.options['x_loc']
        yl = self.options['y_loc']
        ts = self.options['ts']
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']
        mn = self.options['keepout_radius']

        tt, dt = activate(t, ts)

        self.de = {}

        self.de['err_keepoutdist', 'time'] = np.zeros(nn)
        outputs['err_keepoutdist'] = np.zeros(nn)

        for i in range(n_traj):
            x = inputs['x%d' % i]
            y = inputs['y%d' % i]
            dist = np.sqrt((x - xl)**2 + (y - yl)**2)
            dist[np.where(dist < 0.1)] = 0.1

            #dist[np.where(t <= ts)] = mn + 1.0
            f, df = loss(dist, mn)
            f = tt * f
            df = tt * df

            outputs['err_keepoutdist'] += f
            self.de['err_keepoutdist', 'time'] += dt * f

            g1 = df*(x - xl)/(dist)
            if ('err_keepoutdist', 'x%d' % i) not in self.de:
                self.de['err_keepoutdist', 'x%d' % i] = g1
            else:
                self.de['err_keepoutdist', 'x%d' % i] += g1

            g2 = df*(y - yl)/(dist)
            if ('err_keepoutdist', 'y%d' % i) not in self.de:
                self.de['err_keepoutdist', 'y%d' % i] = g2 
            else:
                self.de['err_keepoutdist', 'y%d' % i] += g2 

    def compute_partials(self, inputs, partials):
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']

        for pair in self.de:
            partials[pair] = self.de[pair]

if __name__ == '__main__':
    from openmdao.api import Problem, Group
    nt = 3
    n = 30

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', KeepOut(n_traj = nt, 
                                          num_nodes = n,
                                          x_loc=50.0,
                                          ts=50.0,
                                          keepout_radius=25.0,
                                          y_loc=50.0), promotes=['*'])
    p.setup()

    p['time'] = np.linspace(0,100,n)
    for i in range(nt):
        p['x%d' % i] = np.random.uniform(0.01, 100, size=n)
        p['y%d' % i] = np.random.uniform(0.01, 100, size=n)

    p.run_model()
    p.check_partials(compact_print=True)


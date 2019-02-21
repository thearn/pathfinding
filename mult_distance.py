import numpy as np

from openmdao.api import ExplicitComponent
from itertools import combinations
from numpy import tan


mn = 1.0
min_distance = 500.0


def loss(x, m=500):
    fx = (np.tanh(x - m) - 1) * np.log(x / m)
    df = (-np.tanh(-m + x)**2 + 1)*np.log(x/m) + (np.tanh(-m + x) - 1)/x

    return fx, df

class MDist(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_traj', types=int)
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']
        self.vnames = []
        for i in range(n_traj):
            self.add_input(name='x%d' % i, val=np.zeros(nn))
            self.add_input(name='y%d' % i, val=np.zeros(nn))
            self.vnames.extend(['x%d' % i, 'y%d' % i])

        self.add_output(name='err_dist', val=np.zeros(nn))

        ar = np.arange(nn)
        self.declare_partials('err_dist', self.vnames, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']
        self.de = {}
        outputs['err_dist'] = np.zeros(nn)
        for i, j in combinations([i for i in range(n_traj)], 2):
            x1, y1 = inputs['x%d' % i], inputs['y%d' % i]
            x2, y2 = inputs['x%d' % j], inputs['y%d' % j]
            sum_sq = (x1 - x2)**2 + (y1 - y2)**2

            dist = np.sqrt(sum_sq)
            dist[np.where(dist < mn)] = mn

            f, df = loss(dist, min_distance)

            outputs['err_dist'] += f

            g1 = df*(x1 - x2)/(dist)
            if ('err_dist', 'x%d' % i) not in self.de:
                self.de['err_dist', 'x%d' % i] = g1
            else:
                self.de['err_dist', 'x%d' % i] += g1

            g2 = df*(y1 - y2)/(dist)
            if ('err_dist', 'y%d' % i) not in self.de:
                self.de['err_dist', 'y%d' % i] = g2 
            else:
                self.de['err_dist', 'y%d' % i] += g2 

            g3 = df*(-x1 + x2)/(dist)
            if ('err_dist', 'x%d' % j) not in self.de:
                self.de['err_dist', 'x%d' % j] = g3
            else:
                self.de['err_dist', 'x%d' % j] += g3

            g4 = df*(-y1 + y2)/(dist)
            if ('err_dist', 'y%d' % j) not in self.de:
                self.de['err_dist', 'y%d' % j] = g4
            else:
                self.de['err_dist', 'y%d' % j] += g4


    def compute_partials(self, inputs, partials):
        n_traj = self.options['n_traj']
        nn = self.options['num_nodes']

        for pair in self.de:
            partials[pair] = self.de[pair]



if __name__ == '__main__':
    from openmdao.api import Problem, Group
    nt = 10
    n = 20

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', MDist(n_traj = nt, num_nodes = n), promotes=['*'])
    p.setup()

    for i in range(nt):
        p['x%d' % i] = np.random.uniform(0.01, 500, size=n)
        p['y%d' % i] = np.random.uniform(0.01, 500, size=n)

    p.run_model()
    p.check_partials(compact_print=True)



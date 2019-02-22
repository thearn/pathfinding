import numpy as np

from openmdao.api import ExplicitComponent
from numpy import tan

mn = 0.1

class Schedule(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(name='x_start',
                       val=np.zeros(nn),
                       units='m')

        self.add_input(name='y_start',
                       val=np.zeros(nn),
                       units='m')

        self.add_input(name='x_end',
                       val=np.zeros(nn),
                       units='m')

        self.add_input(name='y_end',
                       val=np.zeros(nn),
                       units='m')

        self.add_input(name='t_departure',
                       val=np.zeros(nn),
                       units='s')

        self.add_input(name='t_arrival',
                       val=np.zeros(nn),
                       units='s')

        # -----
        self.add_input(name='time',
                       val=np.zeros(nn),
                       desc='time vector',
                       units='s')

        self.add_input(name='x',
                       val=np.ones(nn),
                       desc='position x',
                       units='m')

        self.add_input(name='y',
                       val=np.ones(nn),
                       desc='position y',
                       units='m')

        self.add_output(name='err_d',
                       val=np.zeros(nn),
                       desc='schedule defect',
                       units='m')

        self.add_output(name='destination_dist',
                       val=0.0,
                       desc='schedule dist',
                       units='m')

        ar = np.arange(nn)

        self.declare_partials('err_d', ['x', 'y'], rows=ar, cols=ar)
        self.declare_partials('destination_dist', ['x', 'y'])

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        xs, ys = inputs['x_start'], inputs['y_start']
        xe, ye = inputs['x_end'], inputs['y_end']
        td, ta = inputs['t_departure'], inputs['t_arrival']

        t = inputs['time']
        x = inputs['x']
        y = inputs['y']

        pre_flight = np.where(t < td[0])

        post_start = np.where(t >= td[0])

        post_flight = np.where(t > ta[0])

        start_distance = np.sqrt((xs - x)**2 + (ys - y)**2)
        end_distance = np.sqrt((xe - x)**2 + (ye - y)**2)

        outputs['err_d'] = np.zeros(nn)
        outputs['err_d'][pre_flight] = start_distance[pre_flight]
        outputs['err_d'][post_flight] = end_distance[post_flight]

        #end_distance = end_distance * t
        outputs['destination_dist'] = end_distance[post_start].sum()


    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        xs, ys = inputs['x_start'], inputs['y_start']
        xe, ye = inputs['x_end'], inputs['y_end']
        td, ta = inputs['t_departure'], inputs['t_arrival']

        t = inputs['time']
        x = inputs['x']
        y = inputs['y']

        pre_flight = np.where(t < td[0])
        post_start = np.where(t >= td[0])
        post_flight = np.where(t > ta[0])

        diffx = np.zeros(nn)
        diffy = np.zeros(nn)
        start_distance = np.sqrt((x - xs)**2 + (y - ys)**2)
        end_distance = np.sqrt((x - xe)**2 + (y - ye)**2)

        start_distance[np.where(start_distance < mn)] = mn
        end_distance[np.where(end_distance < mn)] = mn

        diffx[pre_flight] = ((x - xs)/start_distance)[pre_flight]
        diffx[post_flight] = ((x - xe)/end_distance)[post_flight]

        diffy[pre_flight] = ((y - ys)/start_distance)[pre_flight]
        diffy[post_flight] = ((y - ye)/end_distance)[post_flight]

        partials['err_d', 'x'] = diffx
        partials['err_d', 'y'] = diffy

        diffx = np.zeros(nn)
        diffy = np.zeros(nn)
        diffx[post_start] = ((x - xe)/end_distance)[post_start]# * t[post_start]
        diffy[post_start] = ((y - ye)/end_distance)[post_start]# * t[post_start]

        partials['destination_dist', 'x'] = diffx
        partials['destination_dist', 'y'] = diffy
        #end_distance[pre_flight] = 0.0
        #partials['destination_dist', 'time'] = end_distance

if __name__ == '__main__':
    from openmdao.api import Problem, Group

    n = 20

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', Schedule(num_nodes = n), promotes=['*'])
    p.setup()

    p['x_start'] = 0.0
    p['y_start'] = 0.0
    p['x_end'] = 100.0
    p['x_end'] = 200.0
    p['t_departure'] = 20.0
    p['t_arrival'] = 120.0

    p['time'] = np.linspace(0, 200, n)
    p['x'] = np.random.uniform(0.01, 300, size=n)
    p['y'] = np.random.uniform(0.01, 300, size=n)

    p.run_model()
    p.check_partials(compact_print=True)



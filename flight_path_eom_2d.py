import numpy as np

from openmdao.api import ExplicitComponent
from numpy import tan



class FlightPathEOM2D(ExplicitComponent):
    """
    Computes the position and velocity equations of motion using
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='vx',
                       val=np.zeros(nn),
                       desc='aircraft velocity magnitude x',
                       units='m/s')

        self.add_input(name='vy',
                       val=np.zeros(nn),
                       desc='aircraft velocity magnitude y',
                       units='m/s')

        # self.add_input(name='T',
        #                val=np.zeros(nn),
        #                desc='thrust',
        #                units='N')

        # self.add_input(name='chi',
        #                val=np.zeros(nn),
        #                desc='heading angle',
        #                units='rad')

        # self.add_input(name='D',
        #                val=np.ones(nn),
        #                desc='drag',
        #                units='N')

        self.add_output(name='x_dot',
                        val=np.zeros(nn),
                        desc='downrange (longitude) velocity',
                        units='m/s')

        self.add_output(name='y_dot',
                        val=np.zeros(nn),
                        desc='crossrange (latitude) velocity',
                        units='m/s')

        self.add_output(name='L_dot',
                        val=np.zeros(nn),
                        desc='roc of total distance travelled',
                        units='m/s')

        # self.add_output(name='v_dot',
        #                 val=np.zeros(nn),
        #                 desc='rate of change of velocity magnitude',
        #                 units='m/s**2')

        self.add_output(name='vt',
                        val=0.0)

        ar = np.arange(nn)

        self.declare_partials('x_dot', 'vx', rows=ar, cols=ar)
        self.declare_partials('y_dot', 'vy', rows=ar, cols=ar)

        self.declare_partials('L_dot', ['vx', 'vy'], rows=ar, cols=ar)

        self.declare_partials('vt', 'vx')
        self.declare_partials('vt', 'vy')
        # self.declare_partials('v_dot', 'D', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        m = 5.0
        #m = inputs['m']
        vx = inputs['vx']
        vy = inputs['vy']
        #T = inputs['T']
        #D = inputs['D']
        #chi = inputs['chi']

        outputs['x_dot'] = vx
        outputs['y_dot'] = vy
        outputs['vt'] = np.sum(vx**2 + vy**2)

        outputs['L_dot'] = np.sqrt(vx**2 + vy**2)


    def compute_partials(self, inputs, partials):
        m = 5.0
        #m = inputs['m']
        vx = inputs['vx']
        vy = inputs['vy']
        #T = inputs['T']
        #D = inputs['D']
        #chi = inputs['chi']


        partials['x_dot', 'vx'] = 1.0
        partials['y_dot', 'vy'] = 1.0

        partials['vt', 'vx'] = 2*vx
        partials['vt', 'vy'] = 2*vy

        v = np.sqrt(vx**2 + vy**2)
        mn = 1e-9
        v[np.where(v < mn)] = mn
        partials['L_dot', 'vx'] = vx/v
        partials['L_dot', 'vy'] = vy/v

if __name__ == '__main__':
    from openmdao.api import Problem, Group

    n = 20

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', FlightPathEOM2D(num_nodes = n), promotes=['*'])
    p.setup()

    p['vx'] = np.random.uniform(0.01, 20, size=n)
    p['vy'] = np.random.uniform(0.01, 20, size=n)

    p.run_model()
    p.check_partials(compact_print=True)


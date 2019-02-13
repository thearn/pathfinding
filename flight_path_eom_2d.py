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

        self.add_input(name='v',
                       val=10*np.ones(nn),
                       desc='aircraft velocity magnitude',
                       units='m/s')

        # self.add_input(name='T',
        #                val=np.zeros(nn),
        #                desc='thrust',
        #                units='N')

        self.add_input(name='chi',
                       val=np.zeros(nn),
                       desc='heading angle',
                       units='rad')

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

        # self.add_output(name='v_dot',
        #                 val=np.zeros(nn),
        #                 desc='rate of change of velocity magnitude',
        #                 units='m/s**2')

        ar = np.arange(nn)

        self.declare_partials('x_dot', 'v', rows=ar, cols=ar)
        self.declare_partials('x_dot', 'chi', rows=ar, cols=ar)

        self.declare_partials('y_dot', 'v', rows=ar, cols=ar)
        self.declare_partials('y_dot', 'chi', rows=ar, cols=ar)

        # self.declare_partials('v_dot', 'T', rows=ar, cols=ar)
        # self.declare_partials('v_dot', 'D', rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        m = 5.0
        #m = inputs['m']
        v = inputs['v']
        #T = inputs['T']
        #D = inputs['D']
        chi = inputs['chi']

        outputs['x_dot'] = v  * np.cos(chi)
        outputs['y_dot'] = v  * np.sin(chi)

        #outputs['v_dot'] = T - D

    def compute_partials(self, inputs, partials):
        m = 5.0
        #m = inputs['m']
        v = inputs['v']
        #T = inputs['T']
        #D = inputs['D']
        chi = inputs['chi']


        partials['x_dot', 'v'] = np.cos(chi)
        partials['x_dot', 'chi'] = -v*np.sin(chi)

        partials['y_dot', 'v'] = np.sin(chi)
        partials['y_dot', 'chi'] = v * np.cos(chi)

        #partials['v_dot', 'T'] = 1.0 
        #partials['v_dot', 'D'] = -1.0 






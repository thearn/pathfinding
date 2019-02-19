import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dymos import load_simulation_results
from plane_ode import n_traj, keepout_radius, x_loc, y_loc, ks_start, n_traj, personal_zone
from itertools import combinations


exp_out = load_simulation_results('sim.db')


data = {}
for i in range(n_traj):
    data[i] = {}
    data[i]['x'] = exp_out.get_values('x%d' % i, units='m').flatten()
    data[i]['y'] = exp_out.get_values('y%d' % i, units='m').flatten()
    data[i]['loc'] = [data[i]['x'][0], data[i]['x'][-1], data[i]['y'][0], data[i]['y'][-1]]


    # data[i]['vx'] = exp_out.get_values('vx%d' % i, units='m/s').flatten()
    # print("vx%d" % i, min(data[i]['vx']), max(data[i]['vx']))


data['t'] = exp_out.get_values('time', units='s').flatten()
print("time", data['t'][-1])
for i, j in combinations([i for i in range(n_traj)], 2):
    dist = exp_out.get_values('distance_%d_%d.dist' % (i, j)).flatten()
    print(i, j, min(dist))

circle_x = []
circle_y = []
r = 4000.
for i in np.linspace(0, 2*np.pi, 1000):
    circle_x.append(r*np.cos(i))
    circle_y.append(r*np.sin(i))

kcircle_x = []
kcircle_y = []
r = keepout_radius
for i in np.linspace(0, 2*np.pi, 1000):
    kcircle_x.append(x_loc + r*np.cos(i))
    kcircle_y.append(y_loc + r*np.sin(i))

n=len(data['t'])
for t in range(len(data['t']))[::-1]:
    fig = plt.figure()
    ax = plt.gca()
    for i in range(n_traj):
        ts = 0
        if ts < 0:
            ts = 0
        if data['t'][t] > ks_start:
            #plt.plot(kcircle_x, kcircle_y, 'k--', linewidth=0.5)   
            circle = plt.Circle((x_loc, y_loc), keepout_radius - personal_zone/2, fill=False)
            ax.add_artist(circle) 
        circle = plt.Circle((0, 0), 4000, fill=False)
        ax.add_artist(circle) 

        circle = plt.Circle((data[i]['x'][t], data[i]['y'][t]), personal_zone/2, fill=False)
        ax.add_artist(circle)

        #plt.plot(data[i]['x'][ts:t+1], data[i]['y'][ts:t+1], 'gray', linewidth=0.1)
        plt.title("t = %f" % data['t'][t])
        #plt.plot(data[i]['x'][t:t+1], data[i]['y'][t:t+1], 'gray')
        c =  data['t'][0:t+1]
        s = np.linspace(0.1,1.0, len(c))
        plt.scatter(data[i]['x'][ts:t+1], data[i]['y'][ts:t+1], marker='o', c=c, s=s, cmap='Greys', alpha=0.5)
        plt.scatter(data[i]['x'][t], data[i]['y'][t], marker='^', cmap='Greens')
        sx, ex, sy, ey = data[i]['loc']
        plt.plot([sx, ex], [sy, ey], 's', markersize=10)
        #plt.xlabel('x')
        #plt.ylabel('y')
    plt.xlim(-5000,5000)
    plt.ylim(-5000,5000)
    plt.tight_layout(pad=1)
    plt.axis('equal')
    fig.savefig('frames/%03d.png' % t, dpi=fig.dpi)



plt.show()

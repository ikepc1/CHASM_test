import numpy as np
import matplotlib.pyplot as plt
from CHASM import CHASM

grid_width = 1.3e3 #meters
grid_side = 10
n_tel = grid_side**2
tel_vectors = np.empty([n_tel,3])
side = np.linspace(-.5*grid_width,.5*grid_width,grid_side)
x,y = np.meshgrid(side,side)
tel_vectors[:,0] = x.flatten()
tel_vectors[:,1] = y.flatten()
tel_vectors[:,2] = np.full(n_tel,0)
# n_tel = 100
# tel_vectors = np.empty([n_tel,3])
# tel_vectors[:,0] = np.linspace(10,1000,100)
# tel_vectors[:,1] = np.full(n_tel,0)
# tel_vectors[:,2] = np.full(n_tel,0)

ch = CHASM(765,8.e7,8.4e4,np.radians(10),'down',tel_vectors,300,600)

plt.ion()
plt.figure()
plt.hist2d(tel_vectors[:,0],tel_vectors[:,1],weights=ch.ng_sum,bins=grid_side)
plt.colorbar()

plt.figure()
hb = plt.hist(ch.counter_time[ch.ng_sum.argmax()],
                      100,
                      weights=ch.ng[ch.ng_sum.argmax()],
                      histtype='step',label='no correction')
hc = plt.hist(ch.counter_time_prime[ch.ng_sum.argmax()],
                      100,
                      weights=ch.ng[ch.ng_sum.argmax()],
                      histtype='step',label='correction')
plt.title('Preliminary Arrival Time Distribution (100 Km from axis)')
# plt.suptitle('(5 degree EE, start height = 0 m, counter height = 525 Km, Xmax = ch.ng_sum.argmax() g/cm^2, Nmax = 1.e8)')
plt.xlabel('Arrival Time [nS]')
plt.ylabel('Number of Cherenkov Photons')
plt.legend()
plt.show()

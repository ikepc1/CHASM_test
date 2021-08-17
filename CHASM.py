from cherenkov_yield import CherenkovYield as cy
# from numba.experimental import jitclass
#
# @jitclass
class CHASM(cy):
    """A class for generating extensive air shower profiles and their Cherenkov
    outputs. The shower can either be a Gaisser Hillas shower or a Griessen
    Shower. The Cartesian origin is at the point where the shower axis intersects
    with the Earth's surface.

    Parameters:
    X_max: depth at shower max (g/cm^2)
    N_max: number of charged particles at X_max
    Lambda: GH Lambda
    X0: Depth of first interaction along axis (g/cm^2)
    theta: Polar angle of the shower axis with respect to vertical. Vertical
    is defined as normal to the Earth's surface at the point where the axis
    intersects with the surface.
    direction: Shower direction, either 'up' for upward going showers, or 'down'
    for downward going showers.
    tel_vectors: user defined array of cartesian telescope locations x,y,z in meters (Numpy Array with shape (N_telescopes,3))
    min_l: minimum accepted Cherenkov wavelength (nm)
    max_l: maximum accepted Cherenkov wavelength (nm)
    split: bool, whether the shower splitting in space methods are called
    """
    def __init__(self,X_array,Nch_array,theta,direction,tel_vectors,min_l,max_l,split = False):
        super().__init__(X_array,Nch_array,theta,direction,tel_vectors,min_l,max_l,split)
        self.split = split

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    tel_vectors = np.empty([100,3])

    theta = np.radians(85)
    phi = 0
    r = 2141673.2772862054

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    tel_vectors[:,0] = np.full(100,x)
    # tel_vectors[:,0] = np.linspace(-1000,1000,100)
    tel_vectors[:,1] = np.linspace(y-100.e3,y+100.e3,100)
    tel_vectors[:,2] = np.full(100,z)

    start_time = time.time()
    ch = CHASM(765,8000.e4,70,-100,np.radians(85),'up',tel_vectors,300,600,split=False)
    end_time = time.time()
    print("Calculations take: %.3f s"%(
        end_time-start_time))

    x = ch.axis_r * np.sin(ch.theta)
    z = ch.axis_r * np.cos(ch.theta)

    arc_angle = 5
    arc = np.linspace(-np.radians(arc_angle),3*np.radians(arc_angle),100)
    x_surf = ch.earth_radius * np.sin(arc)
    z_surf = ch.earth_radius * np.cos(arc) - ch.earth_radius

    x_shower = ch.shower_r * np.sin(ch.theta)
    z_shower = ch.shower_r * np.cos(ch.theta)

    x_width = -ch.shower_rms_w * np.cos(ch.theta)
    z_width = ch.shower_rms_w * np.sin(ch.theta)

    plt.ion()

    plt.figure()
    ax = plt.gca()
    plt.plot(x,z,label='shower axis' )
    plt.plot(x_surf,z_surf,label="Earth's surface")
    plt.scatter(ch.tel_vectors[:,0],ch.tel_vectors[:,2], label='telescopes')
    plt.quiver(x_shower,z_shower,x_width,z_width, angles='xy', scale_units='xy', scale=1,label='shower width')
    plt.quiver(x_shower,z_shower,-x_width,-z_width, angles='xy', scale_units='xy', scale=1)
    plt.plot(x_shower,z_shower,'r',label='Cherenkov region')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.title('Downward Shower 5 degree EE')
    ax.set_aspect('equal')
    plt.grid()

    # Plot Cherenkov distributions of showers that start at different heights
    plt.ion()
    plt.figure()
    plt.plot(ch.tel_vectors[:,1]/1000,ch.ng_sum)
    plt.semilogy()
    plt.xlabel('Counter Position [km from axis]')
    plt.ylabel('Photon Flux [$m^{-2}$]')
    plt.suptitle('Cherenkov Lateral Distribution at Altitude 525 Km')
    plt.title('(5 degree Earth emergence angle)')
    # plt.legend(title = 'Starting Altitude')
    plt.grid()

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
    plt.suptitle('(5 degree EE, start height = 0 m, counter height = 525 Km, Xmax = 500 g/cm^2, Nmax = 1.e8)')
    plt.xlabel('Arrival Time [nS]')
    plt.ylabel('Number of Cherenkov Photons')
    plt.legend()
    plt.show()

    if ch.split:
        plt.ion()
        plt.figure()
        plt.plot(ch.tel_vectors[:,1]/1000,ch.ng_sum)
        plt.plot(ch.tel_vectors[:,1]/1000,ch.split_ng_sum)
        plt.semilogy()
        plt.xlabel('Counter Position [km from axis]')
        plt.ylabel('Photon Flux [$m^{-2}$]')
        plt.suptitle('Cherenkov Lateral Distribution at Altitude 525 Km')
        plt.title('(5 degree Earth emergence angle)')
        # plt.legend(title = 'Starting Altitude')
        plt.grid()

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        totg_to_each = ch.split_ng.sum(axis=0)
        ax.scatter(ch.split_axis[:,0],ch.split_axis[:,1],ch.split_axis[:,2], s = totg_to_each, c = ch.split_shower_nch/ch.split_shower_nch.max(), alpha = 1)
        # ax.scatter(ch.tel_vectors[:,0],ch.tel_vectors[:,1],ch.tel_vectors[:,2])
        ax.set_ylim(-5000,5000)
        # ax.set_xlim(-200,200)
        # ax.set_zlim(0,1000)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        plt.figure()
        hb = plt.hist(ch.split_counter_time[ch.split_ng_sum.argmax()],
                          100,
                          weights=ch.split_ng[ch.split_ng_sum.argmax()],
                          histtype='step',label='no correction')
        hc = plt.hist(ch.split_counter_time_prime[ch.split_ng_sum.argmax()],
                          100,
                          weights=ch.split_ng[ch.split_ng_sum.argmax()],
                          histtype='step',label='correction')
        plt.title('Preliminary Arrival Time Distribution (100 Km from axis)')
        plt.suptitle('(5 degree EE, start height = 0 m, counter height = 525 Km, Xmax = 500 g/cm^2, Nmax = 1.e8)')
        plt.xlabel('Arrival Time [nS]')
        plt.ylabel('Number of Cherenkov Photons')
        plt.legend()
        plt.show()

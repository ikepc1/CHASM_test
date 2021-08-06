import numpy as np
from cherenkov_photon import *
from scipy.integrate import quad
import scipy.constants as spc
from atmosphere import *
from charged_particle import EnergyDistribution, AngularDistribution
import matplotlib.pyplot as plt

# t = 0
# delta = 3e-4
m_e = spc.value('electron mass energy equivalent in MeV')

def theta_to_lg(theta, delta):
    n = 1+delta
    beta = 1 / (n*np.cos(theta))
    value = np.log(m_e/np.sqrt(1-beta**2))
    return np.nan_to_num(value)

def one_direction(theta,f_e,delta):
    l_g = theta_to_lg(theta,delta)
    E_g = np.exp(l_g)
    cherenkov_yield = CherenkovPhoton.cherenkov_yield(E_g,delta)
    return cherenkov_yield * f_e.spectrum(l_g)

def make_CherenkovPhoton_array(n_t=21,min_t=-20.,max_t=20.,
                               n_delta=176,min_lg_delta=-7,max_lg_delta=-3.5,
                               n_theta=321,min_lg_theta=-3,max_lg_theta=0.2):
    t_array = np.linspace(min_t,max_t,n_t)
    delta_array = np.logspace(min_lg_delta,max_lg_delta,n_delta)
    theta_array = np.logspace(min_lg_theta,max_lg_theta,n_theta)
    gg_array = np.empty((n_t,n_delta,n_theta),dtype=float)
    for i,t in enumerate(t_array):
        for j,d in enumerate(delta_array):
            f_e = EnergyDistribution('Tot',t)
            gg = one_direction(theta_array,f_e,d)
            integral = np.trapz(gg * np.sin(theta_array),theta_array)
            if integral != 0:
                gg /= 4 * np.pi * integral
            gg_array[i,j] = gg
    return gg_array,t_array,delta_array,theta_array

gg_array,t_array,delta_array,theta_array = make_CherenkovPhoton_array()
np.savez('one_direction_table.npz',gg_t_delta_theta=gg_array,t=t_array,delta=delta_array,theta=theta_array)
# lgtheta,dlgtheta = np.linspace(-3.,0.2,321,retstep=True)
# theta = 10**lgtheta
# f_e = EnergyDistribution('Tot',t)
# gg = one_direction(theta,f_e,delta)
# gg /= 4 * np.pi * np.trapz(gg * np.sin(theta),theta)
#
# plt.ion()
# plt.figure()
# plt.plot(theta,gg)
# plt.loglog()

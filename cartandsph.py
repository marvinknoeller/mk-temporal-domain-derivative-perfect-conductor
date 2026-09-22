#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:51:15 2024

@author: marvinknoller
"""
import numpy as np
import scipy
def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                         # phi
    return r, elev, az

def sph2cart(r,theta,phi):
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x,y,z

def ymn(theta,phi,n,m):
    my_ymn =np.sign(m)**(abs(m)) * (-1)**(m) * scipy.special.sph_harm(m,n,phi, theta) #the roles of phi and theta are interchanged... 
    return my_ymn

''' 
In these conventions :  r \in [0,infty],
                        phi \in [-pi, pi]
                        theta \in [-pi/2, pi/2]
'''

from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes

addr = get_cython_function_address("scipy.special.cython_special", "lpmv")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
lpmv_fn = functype(addr)

''' Important!:
    The definition of lpmv_fn includes the Condon-Shortley phase, which is the factor (-1)^m
    see https://mathworld.wolfram.com/Condon-ShortleyPhase.html
    This is different than in our community.
'''

FACTORIAL_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@vectorize('int64(int64)')
def vec_factorial(n):
    if n > 20:
        raise ValueError
    return FACTORIAL_TABLE[n]   

@njit
def factorial(n):
    return vec_factorial(n)

@vectorize('float64(float64, float64, float64)')
def vec_lpmv(m, v, x):
    return lpmv_fn(m, v, x)

@njit
def lpmv(m, v, x):
    return vec_lpmv(m, v, x)

@njit
def spherical_harmonics(m, n, azimuthal, polar):
    """
    assume 0 <= m <= n
    """
    x = np.cos(polar)
    val = lpmv(m, n, x)#.astype(np.complex128)
    # print(val)
    val *= np.sqrt((2 * n + 1) / (4.0 * np.pi) * factorial(n - m) / factorial(n + m))
    val *= np.sign(m)**(abs(m)) * (-1)**(m) * np.exp(1j * m * azimuthal) 
    '''
    here, the phase is corrected again so that it fits our definition of Ymn!
    (-1)**m corrects the Condon-Shortley phase and np.sign(m)**abs(m) accounts for negative m's, which 
    are not used here anyway
    '''
    return val




# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm
# # Set values for n and m
# n, m = 1,0

# # Create grid of theta (elevation) and phi (azimuth)
# theta_vals = np.linspace(0, np.pi, 200) - np.pi/2  # Theta from 0 to pi
# phi_vals = np.linspace(0, 2 * np.pi, 200) - np.pi # Phi from 0 to 2pi

# theta, phi = np.meshgrid(theta_vals, phi_vals)

# # Compute spherical harmonics
# Ynm = ymn(theta - np.pi/2, phi+np.pi, n, m)

# # Convert to Cartesian coordinates for plotting
# r = 1#np.abs(Ynm)  # Use the magnitude of the spherical harmonics
# x, y, z = sph2cart(r, theta, phi)

# # Plotting
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # vmin = (np.real(Ynm)/np.max(np.abs(Ynm))).min()
# # vmax = (np.real(Ynm)/np.max(np.abs(Ynm))).max()
# Z = (np.real(Ynm))#/np.max(np.abs(Ynm)))
# # Normalize data for color mapping
# norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
# cmap = cm.get_cmap('jet')
# facecolors=cmap(norm(Z))

# # Create a ScalarMappable for the colorbar
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Only needed for older versions of matplotlib

# # Plot the surface
# # surf = ax.plot_surface(x, y, z, facecolors=plt.cm.jet(np.real(Ynm)/np.max(np.abs(Ynm))), 
# #                        rstride=1, cstride=1, antialiased=True, vmin=vmin, vmax=vmax)
# surf = ax.plot_surface(x,y,z, facecolors=cmap(norm(Z)), rstride=1, cstride=1, antialiased=True)

# fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
# # Label axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # Adjust axis limits
# X,Y,Z = x,y,z
# x_range = [X.min(), X.max()]
# y_range = [Y.min(), Y.max()]
# z_range = [Z.min(), Z.max()]

# # Calculate the range and set equal limits
# max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
# ax.set_xlim([X.mean() - max_range / 2, X.mean() + max_range / 2])
# ax.set_ylim([Y.mean() - max_range / 2, Y.mean() + max_range / 2])
# ax.set_zlim([Z.mean() - max_range / 2, Z.mean() + max_range / 2])
# # Set title
# ax.set_title(f'Spherical Harmonic Y({n},{m})')

# plt.show()

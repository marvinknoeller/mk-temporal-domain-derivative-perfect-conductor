# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:38:50 2023

@author: marvi
"""
import numpy as np

def erronff(Vec):
    N = 10
    N2 = 2*N
    th = np.linspace(np.pi/N, np.pi/N * (N-1),N-1,endpoint=True).T
    Theta = np.ones([N2,1])*th
    w = np.pi/N * np.sqrt(np.sin(Theta))
    w = np.reshape(w.T[:],[180,1])
    err = (np.sum( w.T**2 * np.abs(Vec)**2))**(1/2)
    
    return err

def get_regularization_norm(coeffsreal, coeffsimag, alpha, N):
    mvec1 = np.concatenate([np.sqrt(1 + kk * (kk + 1)) * np.ones(kk+1) for kk in range(0, N + 1)])
    mvec2 = np.concatenate([np.sqrt(1 + kk * (kk + 1)) * np.ones(kk) for kk in range(1, N + 1)])
    phi = -np.sqrt(alpha) * np.concatenate((mvec1 * np.squeeze(coeffsreal), 
                                             mvec2 * np.squeeze(coeffsimag))).reshape(
                                                 int((N+1)**2),1)
                                    
    return np.linalg.norm(phi)

def get_norm_td(Vec, tau):
    # input in the form 3xnum_pointsx(M+1)
    result = np.sqrt(tau * np.sum(np.abs(Vec)**2) )
    return result
    
    
    
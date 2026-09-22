#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:41:18 2024

@author: marvinknoller
"""

import numpy as np

def RKdata(RK):
    """
    Returns coefficients in Butcher notation

    Parameters
    ----------
    RK : int 0, 1, 2, 3 or 4
        RK = 0 for 2 stage Radau IIA
        RK = 1 for 3 stage Radau IIA
        RK = 2 for 3 stage Lobatto IIIC
        RK = 3 for 2 stage Gauss method
        RK = 4 for 3 stage Gauss

    Returns
    -------
    A, b, c : the coefficients of the Runge--Kutta method in the Butcher tableau

    """
    
    if RK == 0:
        # Radau IIA of order 3, stage order 2
        A = np.array([[5/12, -1/12], [3/4, 1/4]])
        b = np.array([3/4, 1/4])
        c = np.array([1/3, 1])
    elif RK == 1:
        # Radau IIA order 5, stage order 3
        A = np.array([[(88-7*np.sqrt(6))/360, (296-169*np.sqrt(6))/1800, (-2+3*np.sqrt(6))/225],
                      [(296+169*np.sqrt(6))/1800, (88+7*np.sqrt(6))/360, (-2-3*np.sqrt(6))/225],
                      [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]])
        b = np.array([(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9])
        c = np.array([(4-np.sqrt(6))/10, (4+np.sqrt(6))/10, 1])
    elif RK == 2:
        # Lobatto IIIC, order 6, stage order 3
        a1 = np.sqrt(5)
        A = np.array([[1/12, -a1/12, a1/12, -1/12],
                      [1/12, 1/4, (10-7*a1)/60, a1/60],
                      [1/12, (10+7*a1)/60, 1/4, -a1/60],
                      [1/12, 5/12, 5/12, 1/12]])
        b = np.array([1/12, 5/12, 5/12, 1/12])
        c = np.array([0, (5-a1)/10, (5+a1)/10, 1])
    elif RK == 3:
        # Gauss method, order 4
        a = np.sqrt(3)/6
        c = np.array([1/2-a, 1/2+a])
        A = np.array([[1/4, 1/4-a], [1/4+a, 1/4]])
        b = np.array([1/2, 1/2])
    elif RK == 4:
        # Gauss method, order 6
        a = np.sqrt(15)/5
        c = np.array([1/2-a/2, 1/2, 1/2+a/2])
        A = np.array([[5/36, 2/9-a/3, 5/36-a/6],
                      [5/36+a*5/24, 2/9, 5/36-a*5/24],
                      [5/36+a/6, 2/9+a/3, 5/36]])
        b = np.array([5/18, 4/9, 5/18])
    else:
        print('Only RK types 0, 1, 2, 3, and 4 are possible.')
        return None, None, None
    
    return A, b, c












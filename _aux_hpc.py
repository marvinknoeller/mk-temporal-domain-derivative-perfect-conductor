#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:50:43 2025

@author: marvinknoller
"""

import numpy as np
import bempp
import numba
from scipy.linalg import lu_solve
from Curl_a_div_op import apply_all_columns

def planewave(f,d,tlag,x,T,M,RK=0,c=0):
    '''
    defines a plane wave 
    Input:
        f: function handle of 1d smooth function
        d: direction of propagation
        tlag: a potential lag (shift to not start at time t=0)
        x: points in space, where you want to have the wave
        normal: the exterior unit normals to the boundary
        T: final time 
        M: number of steps in time
        c: coefficient from the RK method
    Output:
        u: the plane wave at the stages needed for the CQ at the points x
    '''
    if RK == 0:
        t = np.linspace(0,T,M+1);
    elif RK == 1:
        m = c.shape[0]
        t = get_time_points(T,T/M, m, c)
    d = d[:,np.newaxis]
    d = np.tile(d,(1,x.shape[1]))
    direction = np.sum(x*d,axis=0)
    dir_res = direction[:,np.newaxis]
    Ui = np.zeros([direction.size, t.size])
    Ui = f(dir_res - (t-tlag))
    return Ui


def sphericalwave(f,tlag,x,T,M,RK=0,c=0):
    '''
    defines a spherical "curl wave"
    Input:
        f: function handle of 1d smooth function
        tlag: a potential lag (shift to not start at time t=0)
        x: points in space, where you want to have the wave
        normal: the exterior unit normals to the boundary
        T: final time 
        M: number of steps in time
        c: coefficient from the RK method
    Output:
        u: the spherical wave at the stages needed for the CQ at the points x
    '''
    if RK == 0:
        t = np.linspace(0,T,M+1);
    elif RK == 1:
        m = c.shape[0]
        t = get_time_points(T,T/M, m, c)
    Ui = np.zeros([x.shape[1], t.size])
    normx = np.sqrt(np.sum(x**2,0))
    normx = normx[:,np.newaxis]
    Ui = f(t - normx)/(4*np.pi*normx)
    return Ui

def sphericalcurl(f,derf,tlag,x,T,M,RK=0,c=0):
    if RK == 0:
        t = np.linspace(0,T,M+1);
    elif RK == 1:
        m = c.shape[0]
        t = get_time_points(T,T/M, m, c)
    derUi = np.zeros([x.shape[1], t.size, 3])
    normx = np.sqrt(np.sum(x**2,0))
    normx = normx[:,np.newaxis]
    # Ui = f(t - normx)/(4*np.pi*normx)
    derUi =  -x/(4 * np.pi * normx**2) * (derf(t-normx) + f(t-normx)/normx)
    derUi = np.reshape(derUi.T, [x.shape[1], t.size, 3], 'f')
    return derUi

def mollifier(x):
    '''
    Input:
        x: some x values
    Output:
        q: this will be the function, in this case a mollifier
    '''
    Index = x**2<0.999999
    xn = x*Index
    q =  np.exp(-1/(1-xn**2)) * Index
    return q

def dermollifier(x):
    '''
    Input:
        x: some x values
    Output:
        derq: this will be the function, in this case the derivative a mollifier
    '''
    Index = x**2<0.999999
    xn = x*Index
    derq = (-2*xn/((1-xn**2)**2) * np.exp(-1/(1-xn**2)) )*Index
    return derq

@numba.njit
def incomingwave(x,t,direction,example):
    '''
    get the incoming wave for numba computations on the grid
    Input:
        x: some x values
        t: some time values
    Output:
        the modulated Gaussian beam for the computational speed of light c = 1
        on the boundary of the scatterer
    '''
    c0 = 299792458
    if example == 1:
        # ---- CASE 1 : Ballani ----
        f_0 = 2 * 1e8 / c0
        f_bw = 1.5 * 1e8
        sigma = 6.0/(2*np.pi * f_bw) * c0
        sigma_comp = 6.0/(2*np.pi * f_bw)
        t_d, t_d2 = 6 * sigma_comp * c0, 0.0
    elif example == 2:
        # ---- CASE 2 : Monk ----
        sigma = 2.3873 * 1E-8 * c0
        t_d, t_d2 = 2 * 1E-7 * c0, 0.0
        f_0 = 1.2 * 1E8 / c0
    elif example == 3:
        # ---- CASE 3 : Andriulli ----
        f_0 = 1e6 / c0
        sigma = 382 * 1e-9 * c0
        t_d, t_d2 = 10*sigma*3, 10*sigma*3 # here, t_d and t_d2 are delays. Different role than in the other examples
    elif example == 4:
        sigma = 6.3873 * 1E-8 * c0
        t_d, t_d2 = 7 * 1E-7 * c0, 0.0
        f_0 = .1 * 1E8 / c0
    elif example == 5: #for visualization
        sigma = 2. * 1E-8 * c0
        t_d, t_d2 = 2 * 1E-7 * c0, 0.0
        f_0 = .6 * 1E8 / c0
    elif example == 6:
        sigma = 2. * 1E-8 * c0
        t_d, t_d2 = 2 * 1E-7 * c0, 0.0
        f_0 = .4 * 1E8 / c0

    gauss_beam =( np.exp(-1/(2*sigma**2) * (t - (x[0]*direction[0]+x[1]*direction[1]+x[2]*direction[2])-t_d)**2)
                 * np.cos(2*np.pi*f_0*(t-(x[0]*direction[0]+x[1]*direction[1]+x[2]*direction[2])-t_d2)) )
    return gauss_beam

#@numba.njit
def get_time_points(T,tau, m, c):
    '''
    get the time points for the Runge--Kutta CQ
    Input:
        T: final time
        tau: time step size tau = T/M
        m: number of RK stages
        c: coefficient c from RK
    Output:
        ts: the time points needed for the RKCQ
    '''
    N = round(T / tau);
    ts = np.zeros(m * N + 1);   
    for j in range(N):
        for k in range(0,m):
            ts[j * m + k + 1] = j + c[k]
    ts = tau * ts;
    # Note: Tested and fully consistant with Matlab and existing Python code, regarding that we always start with time point 0.
    return ts

def getpointsonslice(extend=3, n1=100, n2=100, slice_to_0='z', origin = np.array([0.0, 0.0, 0.0])):
    xmin, xmax, ymin, ymax = [-extend, extend, -extend, extend]
    plot_grid = np.mgrid[xmin:xmax:n1 * 1j, ymin:ymax:n2 * 1j]
    if slice_to_0 == 'x':
        points = np.vstack((np.zeros(plot_grid[0].size),
                            plot_grid[0].ravel(),
                            plot_grid[1].ravel()))
    elif slice_to_0 == 'y':
        points = np.vstack((plot_grid[0].ravel(),
                            np.zeros(plot_grid[0].size),
                            plot_grid[1].ravel()))
    elif slice_to_0 =='z':      
        points = np.vstack((plot_grid[0].ravel(),
                            plot_grid[1].ravel(),
                            np.zeros(plot_grid[0].size)))
    elif slice_to_0 == 'yz':
        points1 = np.vstack((plot_grid[0].ravel(),
                                np.zeros(plot_grid[0].size),
                                plot_grid[1].ravel()))
        points2 = np.vstack((plot_grid[0].ravel(),
                            plot_grid[1].ravel(),
                            np.zeros(plot_grid[0].size)))
        points = np.concatenate((points1, points2), axis=1)
    return points + np.tile(origin[:,np.newaxis],[1,points.shape[1]])


def trace_Ei_on_scattererJIT(timepoints, Eivars, div_space, curl_space, option = 0):
    """
    Obtain the coefficients that describe E^i x nu on the dofs of the scatterer.
    This is optimized via built in JIT compilation using numba.
    Note that this requires the modified GridFunction in Bempp (still in version 0.3.2)

    Parameters
    ----------
    timepoints : np.array 
        defines the time points
    Eivars : Eivars class
        contains the parameters of the incoming wave
    div_space : bempp.api.space.space.FunctionSpace
        RWG space, not barycentrically refined
    curl_space : bempp.api.space.space.FunctionSpace
        SNC space, not barycentrically refined
    option : int, optional
        defines either a Gaussian beam (0) or a spherical one(1). The default is 0.

    Returns
    -------
    Eicoeff : np.array of floats of dimension #dofs x m*M+1
        coefficients that describe E^i x nu on the dofs of the scatterer.

    """
    A = Eivars.A
    direction = Eivars.direction
    a = Eivars.a
    ex = Eivars.example
    Eicoeff = np.zeros((div_space.global_dof_count,timepoints.shape[0]))
    
    @bempp.api.real_callable(jit=True, parameterized=True)
    def Ei(x, normal, domain_index, result, ti):
        if option == 0:
            incident_field = A * incomingwave(x, ti, direction, example = ex)
        else:
            normx = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            arg = a * (ti - normx)
            Index = arg**2<0.999999
            xn = arg*Index
            molli =  np.exp(-1/(1-xn**2))*Index
            dermolli = a*(-2*xn/((1-xn**2)**2) * np.exp(-1/(1-xn**2)) )*Index #multiplying with a due to inner derivative
            gradui = -x/(4 * np.pi * normx**2) * ( dermolli + molli/normx )
            incident_field = np.cross(gradui, A)
        result[:] = np.cross(incident_field, normal)
    for ii in range(timepoints.shape[0]):
        #print(ii)
        ti = timepoints[ii]
        Eiti = bempp.api.GridFunction(div_space, fun=Ei, dual_space=div_space, 
                                      function_parameters=np.array([ti]))
        Eicoeff[:,ii] = Eiti.coefficients
    return Eicoeff



def RKElCQ_MPI(coeffs, tau, points, M, space, dual_space, A, b, c, local_ell, rank):

    '''
    Perform the first part of the Runge--Kutta convolution quadrature. 
    It is parallelized in order to use it on the supercomputer Roihu.
    Parameters
    ----------
    coeffs : N1 x N2 np.array
        the coefficients of the field on the boundary space x time
    tau : float
        temporal step size
    points : 3 x N_out np.array
        the points in space, where the approx. solution should be obtained
    M : float
        number of time steps 
    space : bempp.space
        Rao-Wilton-Glisson space of the first order
    dual_space : bempp.space
        Scaled Nedelec space of the first order
    A, b, c : np.arrays
        Butcher tableau from the chosen Runge-Kutta method
        
    local_ell : This is the part of the for loop that ...
    rank : ... got

    Returns
    -------
    HtimesnuVec : np.array of complex floats of dimension N1 x m x int(HalfL+1)
        (weird FFT transformed) the coefficients of Hxnu filled only partly due to parallelization using mpi4py
    Es_hatVec : np.array of complex floats of dimension N1 x m x 3 x int(HalfL+1)
        (weird FFT transformed) the Fourier transformed electric field at the points 
    lutuple_list : dictionary
        dictionary of lutuples
    S_list : dictionary
        dictionary of electric single layer potentials
    '''
    # import matplotlib.pyplot as plt
    (N1,N2) = coeffs.shape 
    #N1 is the number of vertices (dofs) of the grid
    #N2 is the number of saved time points, i.e. m*M+1
    #see the above
    m = c.shape[0]    
    # just, when we want to approx. with more than M points
    # L = 2*M
    L = M
    # the radius in the discretization of the Cauchy-Int formula
    if L == 2*M:
        lambda_rad = np.finfo(float).eps**(1/(3*M))
    else:
        lambda_rad = np.finfo(float).eps**(1/(2*M)) 
    # exponents
    jj_exp = np.linspace(0,M-1,M,endpoint=True) #anders als zuvor. warum???
    jj_exp_L = np.linspace(0,L-1,L,endpoint=True)
    # scaled g
    coeffs_fft = np.zeros((N1, m*L), dtype=np.complex128)
    phi_inter = np.zeros((N1, L),  dtype=np.complex128)
    '''Step 1: Post-processing:'''
    for stageInd in range(m):
        phi_inter[:,0:M] = lambda_rad**jj_exp * coeffs[:,np.arange(stageInd+1,m*M+1,m)]
        coeffs_fft[:,np.arange(stageInd,m*L,m)] = np.fft.fft(phi_inter,axis=1)
        
    '''Step 2: The frequency-domain operators need to be applied:'''
    # s_vect = lambda_rad * omega**(-1.0*jj_exp_L)
    s_vect = lambda_rad * np.exp(-1.0j * 2*np.pi * jj_exp_L / L) 
    HalfL = np.ceil(L/2)
    Es_hatVec = np.zeros((points.shape[1] ,m,3,int(HalfL+1)),  dtype=np.complex128)
    HtimesnuVec = np.zeros((N1 ,m,int(HalfL+1)),  dtype=np.complex128)
    if m ==2:
        coeffs_fft1 = coeffs_fft[:, 0::m]
        coeffs_fft2 = coeffs_fft[:, 1::m]
        G = np.concatenate((coeffs_fft1, coeffs_fft2), axis=0)
    elif m == 3:
        coeffs_fft1 = coeffs_fft[:, 0::m]
        coeffs_fft2 = coeffs_fft[:, 1::m]
        coeffs_fft3 = coeffs_fft[:, 2::m]
        G = np.concatenate((coeffs_fft1, coeffs_fft2, coeffs_fft3), axis=0)
    elif m ==4:
        coeffs_fft1 = coeffs_fft[:, 0::m]
        coeffs_fft2 = coeffs_fft[:, 1::m]
        coeffs_fft3 = coeffs_fft[:, 2::m]
        coeffs_fft4 = coeffs_fft[:, 3::m]
        G = np.concatenate((coeffs_fft1, coeffs_fft2, coeffs_fft3, coeffs_fft4), axis=0)
    lutuple_list = {}
    S_list = {}
    for ell in local_ell: #range(0,int(HalfL + 1)): #compute only half of the problems
        #print(ell)
        #print(f"Rank {rank} computing ell={ell}", flush=True)
        deltaMatrix = np.linalg.inv(A + s_vect[ell] * 1.0/(1 - s_vect[ell]) * np.ones((m, 1)) * b.T) # eq. (5.4)
        deltaEigs,T = np.linalg.eig(deltaMatrix / tau) # p. 136 : diagonalize
        # it is: deltaMatrix / tau = T@np.diag(deltaEigs)@np.linalg.inv(T)
        Tinv = np.linalg.inv(T)
        rhsStages = np.matmul(np.reshape(G[:, ell], (N1, m),order='F'),Tinv.T) #vertauscht aber ok
        lhsStagesVec = np.zeros((points.shape[1],m,3),  dtype=np.complex128)
        lhsStagesPhi = np.zeros((N1,m),  dtype=np.complex128)
        for stageInd in range(m):
            '''Solve the scattering problem start'''
            grid_fun = bempp.api.GridFunction(space=space, 
                                              coefficients = deltaEigs[stageInd]*rhsStages[:,stageInd],
                                              dual_space=dual_space)
            
            # l2norm = grid_fun.l2_norm()
            # print(l2norm)
            if np.abs(deltaEigs[stageInd])**3*np.exp(-deltaEigs[stageInd].real) > 1e-13:
            # if l2norm > 1e-12:
                V = deltaEigs[stageInd] * bempp.api.operators.boundary.maxwell.electric_field(
                    space, space, dual_space, 1.0j * deltaEigs[stageInd])
                S = bempp.api.operators.potential.maxwell.electric_field(
                    space, points, 1.0j * deltaEigs[stageInd])
                '''Solve the scattering problem end'''
                lutuple = bempp.api.linalg.direct_solvers.compute_lu_factors(V)
                lutuple_list[(ell,stageInd)] = lutuple
                S_list[(ell,stageInd)] = S
                # phi = lu(V,grid_fun, lu_factor=lutuple) # replaced by following two lines
                vec = grid_fun.projections(V.dual_to_range)
                phicoeff = lu_solve(lutuple, vec)
                phi = bempp.api.GridFunction(space, coefficients=phicoeff)
                result = S.evaluate(phi).T # bis hier: phi = - V^{-1}(s) \gamma_t E^i
                lhsStagesVec[:,stageInd,:] = result
                lhsStagesPhi[:,stageInd]= phi.coefficients # Hxnu = -phi ! ### UNSURE!
            else:
                print('skipped.')
        HtimesnuVec[:,:,ell] = np.einsum('ij,jl->il', lhsStagesPhi, T.T)
        Es_hatVec[:,:,:,ell] = np.einsum('ijk,jl->ilk', lhsStagesVec, T.T)
        
    return HtimesnuVec, Es_hatVec, lutuple_list, S_list
        

def RKElCQ_solo(coeffs, points, M, c, HtimesnuVec, Es_hatVec):
    """
    Perform the second part of the Runge--Kutta convolution quadrature. 
    Parameters
    ----------
    coeffs : N1 x N2 np.array
        the coefficients of the field on the boundary space x time
    points : 3 x N_out np.array
        the points in space, where the approx. solution should be obtained
    M : float
        number of time steps 
    c : np.arrays
        c from the Butcher tableau from the chosen Runge-Kutta method
    HtimesnuVec : np.array of complex floats of dimension N1 x m x int(HalfL+1)
        (weird FFT transformed) the coefficients of Hxnu filled only partly due to parallelization using mpi4py
    Es_hatVec : np.array of complex floats of dimension N1 x m x 3 x int(HalfL+1)
        (weird FFT transformed) the Fourier transformed electric field at the points 

    Returns
    -------
    Es_sol : np.array of floats of dimension N x m*M+1 x 3
        The scattered field at the points
    Htimesnu_solL : np.array of floats of dimension N1 x m*M+1
        coefficients of Hxnu on the dofs on the boundary

    """
    
    (N1,N2) = coeffs.shape 
    #N1 is the number of vertices (dofs) of the grid
    #N2 is the number of saved time points, i.e. m*M+1
    #see the above
    m = c.shape[0]    
    # just, when we want to approx. with more than M points
    # L = 2*M
    L = M
    # the radius in the discretization of the Cauchy-Int formula
    if L == 2*M:
        lambda_rad = np.finfo(float).eps**(1/(3*M))
    else:
        lambda_rad = np.finfo(float).eps**(1/(2*M))
    HalfL = np.ceil(L/2)
    # exponents
    jj_exp_L = np.linspace(0,L-1,L,endpoint=True)
    Htimesnu_hat_half = np.reshape(HtimesnuVec, (N1, int((HalfL + 1) * m)),
                              order='F') #reshaping in python works different than in Matlab
    Es_hat_half = np.reshape(Es_hatVec.transpose(0,2,1,3), (points.shape[1], 3, int((HalfL + 1) * m)),
                              order='F') #reshaping in python works different than in Matlab
    # Htimesnu_hat_half = Htimesnu_hat_half.transpose(0,2,1)
    Es_hat_half = Es_hat_half.transpose(0,2,1)
    freqInd = np.arange(HalfL + 1, L)
    sourceIndices = np.add.outer(m * (L - freqInd), np.arange(m)).flatten()
    sourceIndicesInt =  [int(x) for x in sourceIndices]
    targetIndices = np.add.outer(freqInd * m, np.arange(m)).flatten()
    targetIndicesInt =  [int(x) for x in targetIndices]
    
    Htimesnu_hat = np.zeros((N1,int(m*L)),  dtype=np.complex128)
    Htimesnu_hat[:,:int((HalfL+1)*m)] = Htimesnu_hat_half
    Htimesnu_hat[:, targetIndicesInt] = np.conj(Htimesnu_hat[:, sourceIndicesInt]) #passt denke ich
    
    Es_hat = np.zeros((points.shape[1],int(m*L),3),  dtype=np.complex128)
    Es_hat[:,:int((HalfL+1)*m),:] = Es_hat_half
    Es_hat[:, targetIndicesInt, :] = np.conj(Es_hat[:, sourceIndicesInt, :]) #passt denke ich
    
    '''Step 3: Postprocessing'''
    Htimesnu_solL = np.zeros((N1, m * L))
    Es_solL = np.zeros((points.shape[1], m * L, 3))
    miss_factor = lambda_rad**(-jj_exp_L)
    for stageInd in range(m):
        Htimesnu_inter = np.fft.ifft(Htimesnu_hat[:,np.arange(stageInd,m*L,m)],axis=1)
        Htimesnu_solL[:,np.arange(stageInd,m*L,m)] = np.real( 
            miss_factor[np.newaxis, :] * Htimesnu_inter)
        
        Es_inter = np.fft.ifft(Es_hat[:,np.arange(stageInd,m*L,m),:],axis=1)
        Es_solL[:,np.arange(stageInd,m*L,m), :] = np.real( 
            miss_factor[np.newaxis, :, np.newaxis] * Es_inter)
    Htimesnu_sol = np.zeros((N1,m*M+1))
    Htimesnu_sol[:,1:] = Htimesnu_solL[:,:m*M]
    Es_sol = np.zeros((points.shape[1],m*M+1,3))
    Es_sol[:,1: ,:] = Es_solL[:,:m*M,:]
    
    return Es_sol, Htimesnu_sol 

"""---------------------------------- SHAPE DERIVATIVE RELATED THINGS START FROM HERE -------------------------------------------"""

def RKElCQ_MPI_derivative(
        Htimesnu,
        h_nu,
        tau,
        points,
        M,
        div_space,
        curl_space,
        scalar_bem_space,
        A,
        b,
        c,
        lutuple_list,
        S_list,
        local_ell,
        MPI,
        rank,
        comm
        ):
    """
    Perform the Runge--Kutta convolution quadrature for the time-dependent Maxwell domain derivative.
    It is parallelized in order to use it on the supercomputer Roihu.

    Parameters
    ----------
    Htimesnu : np.array of floats of dimension points.shape[1] x m*M+1 x 3
        H x nu on the dofs at all the time points
    h_nu : bempp.api.GridFunction
        Contains all perturbations of the scattering object
    tau : float
        Temporal step size T/M
    evaluation_points : np.array of floats 3xM
        the points, in which you want to have the scattered field in the end
    M : int
        number of time steps
    div_space : bempp.api.space.space.FunctionSpace
        RWG space, not barycentrically refined
    curl_space : bempp.api.space.space.FunctionSpace
        SNC space, not barycentrically refined
    scalar_bem_space : bempp.api.space.space.FunctionSpace
        P1 scalar valued space
    A, b, c : np.arrays
        Butcher tableau from the chosen Runge-Kutta method
    lutuple_list : dictionary
        dictionary of lutuples
    S_list : dictionary
        dictionary of electric single layer potentials
    local_ell : This is the part of the for loop that ...
        rank : ... got
    MPI, rank, comm : MPI parameters

    Returns
    -------
    Es_sol : np.array of floats of dimension #points x m*M+1 x 3 x (N+1)^2
        the domain derivative at the points for all perturbations h at all points

    """
    
    # import matplotlib.pyplot as plt
    (N1, N2) = Htimesnu.shape 
    #N1 is the number of vertices (dofs) of the grid
    #N2 is the number of saved time points, i.e. m*M+1
    # (no_dofs_on_scalar_bem_space, N2) = Div_trace_H_all.shape
    h_nu_coeff = h_nu.coefficients
    (no_dofs_on_scalar_bem_space, np1squared) = h_nu_coeff.shape #np1squared = (N+1)^2
    #see the above
    m = c.shape[0]    
    # just, when we want to approx. with more than M points
    # L = 2*M
    L = M
    # the radius in the discretization of the Cauchy-Int formula
    if L == 2*M:
        lambda_rad = np.finfo(float).eps**(1/(3*M))
    else:
        lambda_rad = np.finfo(float).eps**(1/(2*M))
    # exponents
    jj_exp = np.linspace(0,M-1,M,endpoint=True) #anders als zuvor. warum???
    jj_exp_L = np.linspace(0,L-1,L,endpoint=True)
    
    Htimesnu_all_fft = np.zeros((N1, m*L), dtype=np.complex128)
    Htimesnu_all_inter = np.zeros((N1, L),  dtype=np.complex128)
    '''Step 1: Post-processing:'''
    for stageInd in range(m):
        Htimesnu_all_inter[:,0:M] = lambda_rad**jj_exp * Htimesnu[:,np.arange(stageInd+1,m*M+1,m)]
        Htimesnu_all_fft[:,np.arange(stageInd,m*L,m)] = np.fft.fft(Htimesnu_all_inter,axis=1)
        
    '''Step 2: The frequency-domain operators need to be applied:'''
    s_vect = lambda_rad * np.exp(-1.0j * 2*np.pi * jj_exp_L / L)
    HalfL = np.ceil(L/2)
    Eprime_all_h_hatVec = np.zeros((points.shape[1] ,m ,3, int(HalfL+1), np1squared),  dtype=np.complex128)
    if m ==2:
        Htimesnu_all_fft1 = Htimesnu_all_fft[:, 0::m]
        Htimesnu_all_fft2 = Htimesnu_all_fft[:, 1::m]
        Htimesnu_all = np.concatenate((Htimesnu_all_fft1, Htimesnu_all_fft2), axis=0)
        
    elif m == 3:
        Htimesnu_all_fft1 = Htimesnu_all_fft[:, 0::m]
        Htimesnu_all_fft2 = Htimesnu_all_fft[:, 1::m]
        Htimesnu_all_fft3 = Htimesnu_all_fft[:, 2::m]
        Htimesnu_all = np.concatenate((Htimesnu_all_fft1, Htimesnu_all_fft2, Htimesnu_all_fft3), axis=0)

    elif m ==4:
        Htimesnu_all_fft1 = Htimesnu_all_fft[:, 0::m]
        Htimesnu_all_fft2 = Htimesnu_all_fft[:, 1::m]
        Htimesnu_all_fft3 = Htimesnu_all_fft[:, 2::m]
        Htimesnu_all_fft4 = Htimesnu_all_fft[:, 3::m]
        Htimesnu_all = np.concatenate((Htimesnu_all_fft1, Htimesnu_all_fft2, Htimesnu_all_fft3, Htimesnu_all_fft4), axis=0)

    for ell in local_ell: #compute only half of the problems
        deltaMatrix = np.linalg.inv(A + s_vect[ell] * 1.0/(1 - s_vect[ell]) * np.ones((m, 1)) * b.T) # eq. (5.4)
        deltaEigs,T = np.linalg.eig(deltaMatrix / tau) # p. 136 : diagonalize
        # it is: deltaMatrix / tau = T@np.diag(deltaEigs)@np.linalg.inv(T)
        Tinv = np.linalg.inv(T)
        rhsStages_Htimesnu = np.matmul(np.reshape(Htimesnu_all[:, ell],
                                                  (N1, m),order='F'),Tinv.T) #vertauscht aber ok
        lhsStagesVec = np.zeros((points.shape[1],m,3,np1squared),  dtype=np.complex128)
        
        for stageInd in range(m):
            '''Solve the scattering problem start'''
            ''' In the grid_functions we multiply the coefficients with s, since this was also done for the operator V(s)'''
            grid_fun_Htimesnu = bempp.api.GridFunction(space=div_space, 
                                              coefficients = deltaEigs[stageInd]*rhsStages_Htimesnu[:,stageInd],
                                              dual_space=curl_space)
            
            if np.abs(deltaEigs[stageInd])**3*np.exp(-deltaEigs[stageInd].real) > 1e-13:
                '''Solve the scattering problem end'''
                lutuple = lutuple_list[(ell,stageInd)]
                S = S_list[(ell,stageInd)]
                ''' Now create the right hand side'''

                out = apply_all_columns(div_space, (1.0 / deltaEigs[stageInd])*h_nu_coeff,  grid_fun_Htimesnu, scalar_bem_space, "snc")   # (ndof, (N+1)**2)
                Curl_hnu_Enu = out
                        
                def rhs_T2(phi, h_nu, X, s):
                    """b_i = s * <h_nu (nu x phi), x_i>,  x_i in X. Returns projections on X's global dofs."""
                    from bempp.api.integration.triangle_gauss import rule as gauss_quad_rule
                
                    pts, w = gauss_quad_rule(bempp.api.GLOBAL_PARAMETERS.quadrature.regular)
                    c_X = X
                
                    grid = c_X.grid
                    probe = h_nu.evaluate(int(c_X.support_elements[0]), pts)
                    ncols = probe.shape[2] if probe.ndim == 3 else None
                    b = np.zeros((c_X.grid_dof_count,) if ncols is None
                                 else (c_X.grid_dof_count, ncols), dtype=complex)
                
                    for e in c_X.support_elements:
                        nu = grid.normals[e].reshape((3, 1))
                        ie = grid.integration_elements[e]
                        nxp = np.cross(nu, phi.evaluate(e, pts), axis=0)   # (3, nq)
                        h = h_nu.evaluate(e, pts)                          # (1, nq) or (1, nq, ncols)
                        basis = c_X.evaluate(e, pts)                       # multipliers already applied
                        dofs = c_X.local2global[e]
                        if ncols is None:
                            integrand = nxp * h
                            for j in range(3):
                                b[dofs[j]] += np.sum(np.sum(integrand * basis[:, j, :], axis=0) * w * ie)
                        else:
                            integrand = nxp[:, :, None] * h
                            for j in range(3):
                                contracted = np.sum(integrand * basis[:, j, :][:, :, None], axis=0)
                                b[dofs[j], :] += np.sum(contracted * (w * ie)[:, None], axis=0)
                
                    return s * b
                
                ik_hnu_tangH = bempp.api.GridFunction(
                    div_space, projections=rhs_T2(grid_fun_Htimesnu, h_nu, div_space, deltaEigs[stageInd]), dual_space=div_space
                )
                
                ik_hnu_tangH = ik_hnu_tangH.projections(curl_space)
                # Now assemble the rhs as a GridFunction
                rhs_h = ik_hnu_tangH + Curl_hnu_Enu
                # Solve via single layer potential ansatz
                x = lu_solve(lutuple, rhs_h)
                Phi = bempp.api.GridFunction(div_space, coefficients=x)    # coefficients=, not projections=
                
                for hind in range(0,np1squared):
                    phi = bempp.api.GridFunction(
                        div_space, dual_space=curl_space, coefficients=Phi.coefficients[:,hind])
                    lhsStagesVec[:,stageInd,:,hind] = S.evaluate(phi).T

            else:
                print('skipped.')
        Eprime_all_h_hatVec[:,:,:,ell,:] = np.einsum('ijkm,jl->ilkm', lhsStagesVec, T.T)
        
    fullEprime = np.zeros_like(Eprime_all_h_hatVec) if rank == 0 else None
    comm.Reduce(Eprime_all_h_hatVec, fullEprime, op=MPI.SUM, root=0)
    comm.Barrier()
    if rank==0:
        Es_hat_half = np.reshape(fullEprime.transpose(0,2,1,3,4), (points.shape[1], 3, int((HalfL + 1) * m), np1squared),
                                  order='F') #reshaping in python works different than in Matlab
        # Htimesnu_hat_half = Htimesnu_hat_half.transpose(0,2,1)
        Es_hat_half = Es_hat_half.transpose(0,2,1,3)
        freqInd = np.arange(HalfL + 1, L)
        sourceIndices = np.add.outer(m * (L - freqInd), np.arange(m)).flatten()
        sourceIndicesInt =  [int(x) for x in sourceIndices]
        targetIndices = np.add.outer(freqInd * m, np.arange(m)).flatten()
        targetIndicesInt =  [int(x) for x in targetIndices]
        
        
        Es_hat = np.zeros((points.shape[1],int(m*L),3,np1squared),  dtype=np.complex128)
        Es_hat[:,:int((HalfL+1)*m),:,:] = Es_hat_half
        Es_hat[:, targetIndicesInt, :, :] = np.conj(Es_hat[:, sourceIndicesInt, :, :]) 
        
        '''Step 3: Postprocessing'''
        Es_solL = np.zeros((points.shape[1], m * L, 3, np1squared))
        miss_factor = lambda_rad**(-jj_exp_L)
        for stageInd in range(m):
            
            Es_inter = np.fft.ifft(Es_hat[:,np.arange(stageInd,m*L,m),:,:],axis=1)
            Es_solL[:,np.arange(stageInd,m*L,m), :, :] = np.real( 
                miss_factor[np.newaxis, :, np.newaxis, np.newaxis] * Es_inter)

        Es_sol = np.zeros((points.shape[1],m*M+1,3,np1squared))
        Es_sol[:,1: ,:, :] = Es_solL[:,:m*M,:, :]
        
    else:
        Es_sol = None
        
    return Es_sol 


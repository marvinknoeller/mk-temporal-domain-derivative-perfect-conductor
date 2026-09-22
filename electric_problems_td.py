#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:38:56 2025

@author: marvinknoller
"""
import bempp.api
import RK_Data
import _aux_hpc
import numpy as np
import gridfun_operations as gfo
import correct_space as corr
import numba
from cartandsph import spherical_harmonics

def efie_td(scatterer_grid,
            Eivars,
            points,
            evaluation_points,
            idx,
            T,
            M,
            RK,
            MPI,
            rank,
            comm,
            subinterval,
            visualization=0):
    """
    
    Parameters
    ----------
    scatterer_grid : bempp.api.grid.grid.Grid
        the given grid of the scatterer. Could also come from starshaped.Star.grid
    Eivars : Eivars class
        contains the parameters of the incoming wave
    points : np.array of floats 3xN
        all points, eventually including points that lie inside the scatterer
    evaluation_points : np.array of floats 3xM
        the points, in which you want to have the scattered field in the end
            points and evaluation_points can coincide!
    idx : np.array of bools
        index set for points that lie outside the scatterer
    T : float
        final time
    M : int
        number of time steps
    RK : int
        determines the Runge-Kutta method
    MPI : mpi4py package
        imported mpi4py package that has already been loaded
    rank : int
        rank of thread
    comm : MPI.COMM_WORLD
        comm = MPI.COMM_WORLD
    subinterval : np.array of int size depends
        this is the subinterval for each rank
    visualization : int either 0 or 1, optional
        do you want to save the waves for later visualization? The default is 0.

    Returns
    -------
    div_space : bempp.api.space.space.FunctionSpace
        RWG space, not barycentrically refined
    curl_space : bempp.api.space.space.FunctionSpace
        SNC space, not barycentrically refined
    Es_on_plane : np.array of floats of dimension points.shape[1] x M+1 x 3
        the scatted electric field on the points. None, when inside.
    Htimesnu_all : np.array of floats of dimension points.shape[1] x m*M+1 x 3
        H x nu on the dofs at all the time points
    lutuple_list : dictionary
        dictionary of lutuples
    S_list : dictionary
        dictionary of electric single layer potentials
    eionplane : np.array of dimension points.shape[1] x M+1 x 3
        the incoming electric field on the points. None, when inside
    eitimesnuonscat : np.array of floats of dimension dofs x M+1 x 3
        Ei x nu on the boundary of the object for all time points
    centroids : np.array of floats of dimension V x 3
        the centroids of the grid
    norm_on_point : np.array of floats of dimension M+1
        the norm of the scattered wave at some sample point somewhere
    normHtimesnu : np.array of floats of dimension M+1
        the norm of Hxnu at some sample point on the grid somewhere

    """
    div_space = bempp.api.function_space(scatterer_grid,'RWG',0)
    curl_space = bempp.api.function_space(scatterer_grid,'SNC',0)
    A,b,c = RK_Data.RKdata(RK)
    tau = T/M # time step size
    
    if rank == 0:
        m = c.shape[0]
        timepoints = _aux_hpc.get_time_points(T, tau, m, c)
        
        ''' Get the coefficients of the trace of the incoming field on the scatterer'''
        eicoeff = _aux_hpc.trace_Ei_on_scattererJIT(timepoints, Eivars, div_space, curl_space)
        # this is for visualization purposes!
        if visualization:
            eionplane = _aux_hpc.Eionplane(timepoints, Eivars, points, RK=RK, c=c)
            eicoeffvis = eicoeff[:,np.arange(0,m*M+1,m)]
            # evaluate E^i x nu on the scatterer
            eitimesnuonscat = np.zeros((scatterer_grid.elements.shape[1],M+1,3))
            for ii in range(M+1):
                funfun = bempp.api.GridFunction(div_space, coefficients = eicoeffvis[:,ii])
                eitimesnuonscat[:,ii,:] = funfun.evaluate_on_element_centers().T
        else: 
            eitimesnuonscat = None
        
    else:
        eionplane = None
        eicoeff = None
        eitimesnuonscat = None

    
    eicoeff = comm.bcast(eicoeff, root=0)
    ## die H's sind wichtig! Gebraucht in shape derivative... 
    HtimesnuVecloc, Es_hatVecloc, lutuple_list, S_list = _aux_hpc.RKElCQ_MPI(coeffs=-eicoeff,
                                       tau=tau,
                                       points=evaluation_points,
                                       M = M,
                                       space=div_space,
                                       dual_space=curl_space,
                                       A=A,b=b,c=c,
                                       local_ell=subinterval, rank=rank)
    comm.Barrier()
    HtimesnuVec = np.zeros_like(HtimesnuVecloc) if rank == 0 else None
    comm.Reduce(HtimesnuVecloc, HtimesnuVec, op=MPI.SUM, root=0)
    Es_hatVec = np.zeros_like(Es_hatVecloc) if rank == 0 else None
    comm.Reduce(Es_hatVecloc, Es_hatVec, op=MPI.SUM, root=0)
    comm.Barrier()
    
    if rank == 0:
        ''' The scattered electric field is the result of the convolution quadrature'''
        # Htimesnu = np.zeros((scatterer.grid.elements.shape[1],M+1,3))
        Es_on_plane = np.zeros((points.shape[1],m*M+1, 3))
        Es_on_plane[:] = np.nan
        Es_on_plane[idx,:,:], Htimesnu_all = _aux_hpc.RKElCQ_solo(
            coeffs=-eicoeff,
            points=evaluation_points,
            M=M,
            c=c,
            HtimesnuVec=HtimesnuVec,
            Es_hatVec=Es_hatVec)
        ''' Get the fields at the original time points '''
        Htimesnu = Htimesnu_all[:,np.arange(0,m*M+1,m)]
        Es_on_plane = Es_on_plane[:,np.arange(0,m*M+1,m),:]
        if visualization:
            eionplane = eionplane[:,np.arange(0,m*M+1,m),:]
            ''' Export the variables for better visualization in Matlab '''
            centroids = scatterer_grid.centroids
            normHtimesnu = np.zeros((M+1))
            norm_on_point = np.zeros(M+1)
            point_idx = 20
            for ii in range(M+1):
                funfun = bempp.api.GridFunction(div_space, coefficients = Htimesnu[:,ii])
                F = funfun.evaluate_on_element_centers()
                norm_on_point[ii] = np.linalg.norm(F[:,point_idx])
                normHtimesnu[ii] = funfun.l2_norm()
        else:
            eionplane = None
            centroids = None
            norm_on_point = None
            normHtimesnu = None
            
    else:
        Es_on_plane = None
        Htimesnu = None
        Htimesnu_all = None
        centroids = None
        norm_on_point = None
        normHtimesnu = None
        
    Es_on_plane = comm.bcast(Es_on_plane, root=0)
    Htimesnu = comm.bcast(Htimesnu, root=0)
    Htimesnu_all = comm.bcast(Htimesnu_all, root=0)
    
    if visualization:
        eionplane = comm.bcast(eionplane, root=0)
        eitimesnuonscat = comm.bcast(eitimesnuonscat, root=0)
        centroids = comm.bcast(centroids, root=0)
        norm_on_point = comm.bcast(norm_on_point, root=0)
        normHtimesnu = comm.bcast(normHtimesnu, root=0)
            
    return (div_space,
            curl_space,
            Es_on_plane,
            Htimesnu_all,
            lutuple_list,
            S_list,
            eionplane,
            eitimesnuonscat,
            centroids,
            norm_on_point,
            normHtimesnu)

def efie_td_ff(scatterer_grid,
               Eivars,
               farfieldpoints,
               T,
               M,
               RK,
               MPI,
               rank,
               comm,
               subinterval):
    """

    Parameters
    ----------
    scatterer_grid : bempp.api.grid.grid.Grid
        the given grid of the scatterer. Could also come from starshaped.Star.grid
    Eivars : Eivars class
        contains the parameters of the incoming wave
    farfieldpoints : np.array of floats of dimension 3 x N
        DESCRIPTION.
    T : float
        final time
    M : int
        number of time steps
    RK : int
        determines the Runge-Kutta method
    MPI : mpi4py package
        imported mpi4py package that has already been loaded
    rank : int
        rank of thread
    comm : MPI.COMM_WORLD
        comm = MPI.COMM_WORLD
    subinterval : np.array of int size depends
        this is the subinterval for each rank

    Returns
    -------
    div_space : bempp.api.space.space.FunctionSpace
        RWG space, not barycentrically refined
    curl_space : bempp.api.space.space.FunctionSpace
        SNC space, not barycentrically refined
    ff : np.array of floats of dimension N x M+1 x 3
        the time dependent far field at the farfieldpoints
    Htimesnu_all : np.array of floats of dimension points.shape[1] x m*M+1 x 3
        H x nu on the dofs at all the time points
    lutuple_list : dictionary
        dictionary of lutuples

    """

    div_space = bempp.api.function_space(scatterer_grid,'RWG',0)
    curl_space = bempp.api.function_space(scatterer_grid,'SNC',0)
    A,b,c = RK_Data.RKdata(RK)
    tau = T/M # time step size
    
    if rank == 0:
        m = c.shape[0]
        timepoints = _aux_hpc.get_time_points(T, tau, m, c)
        
        ''' Get the coefficients of the trace of the incoming field on the scatterer'''
        eicoeff = _aux_hpc.trace_Ei_on_scattererJIT(timepoints, Eivars, div_space, curl_space)
        # this is for visualization purposes!
        
    else:
        eicoeff = None

    
    eicoeff = comm.bcast(eicoeff, root=0)
    ## die H's sind wichtig! Gebraucht in shape derivative... 
    HtimesnuVecloc, Es_hatVecloc, lutuple_list, S_list = _aux_hpc.RKElCQ_MPI_ff(coeffs=-eicoeff,
                                       tau=tau,
                                       points=farfieldpoints,
                                       M = M,
                                       space=div_space,
                                       dual_space=curl_space,
                                       A=A,b=b,c=c,
                                       local_ell=subinterval, rank=rank)
    comm.Barrier()
    HtimesnuVec = np.zeros_like(HtimesnuVecloc) if rank == 0 else None
    comm.Reduce(HtimesnuVecloc, HtimesnuVec, op=MPI.SUM, root=0)
    Es_hatVec = np.zeros_like(Es_hatVecloc) if rank == 0 else None
    comm.Reduce(Es_hatVecloc, Es_hatVec, op=MPI.SUM, root=0)
    comm.Barrier()
    
    if rank == 0:
        ''' The scattered electric field is the result of the convolution quadrature'''
        # Htimesnu = np.zeros((scatterer.grid.elements.shape[1],M+1,3))
        ff = np.zeros((farfieldpoints.shape[1],m*M+1, 3))
        ff[:] = np.nan
        ff, Htimesnu_sol = _aux_hpc.RKElCQ_solo(
            coeffs=-eicoeff,
            points=farfieldpoints,
            M=M,
            c=c,
            HtimesnuVec=HtimesnuVec,
            Es_hatVec=Es_hatVec)
        ''' Get the fields at the original time points '''
        ff = ff[:,np.arange(0,m*M+1,m),:]
            
    else:
        ff = None
        
    ff = comm.bcast(ff, root=0)
            
    return (div_space,
            curl_space,
            ff,
            lutuple_list,
            S_list)
    
    

def solve_maxwell_domain_derivative_pc_td(scatterer,
                                          div_space,
                                          curl_space,
                                          Htimesnu,
                                          N,
                                          T,
                                          M,
                                          evaluation_points,
                                          RK,
                                          lutuple_list,
                                          S_list,
                                          MPI,
                                          rank,
                                          comm,
                                          size,
                                          subinterval):
    """

    Parameters
    ----------
    scatterer : starshaped.Star
        a starshaped object
    div_space : bempp.api.space.space.FunctionSpace
        RWG space, not barycentrically refined
    curl_space : bempp.api.space.space.FunctionSpace
        SNC space, not barycentrically refined
    Htimesnu : np.array of floats of dimension points.shape[1] x m*M+1 x 3
        H x nu on the dofs at all the time points
    N : int
        the number of deformations is (N+1)^2
    T : float
        final time
    M : int
        number of time steps
    evaluation_points : np.array of floats 3xM
        the points, in which you want to have the scattered field in the end
    RK : int
        determines the Runge-Kutta method
    lutuple_list : dictionary
        dictionary of lutuples
    S_list : dictionary
        dictionary of electric single layer potentials
    MPI : mpi4py package
        imported mpi4py package that has already been loaded
    rank : int
        rank of thread
    comm : MPI.COMM_WORLD
        comm = MPI.COMM_WORLD
    size : comm.Get_size()
        size = comm.Get_size()
    subinterval : np.array of int size depends
        this is the subinterval for each rank

    Returns
    -------
    jacobian : np.array of floats of dimension evaluation_points.shape[1] x M+1 x 3 x np1squared
        This will be the jacobian that is required for the shape optimization

    """
    
    tau = T/M
    # Initialize Frechet derivative
    numpoints = evaluation_points.shape[1]
    (N1, mMp1) = Htimesnu.shape  # mMp1 = m*M+1 AND M=L N1 = no. of dofs
    A,b,c = RK_Data.RKdata(RK)
    m = c.shape[0]
    M = (mMp1 - 1)//m
    jacobian = np.zeros([3, numpoints, M, (N+1)**2])
    scalar_bem_space = bempp.api.function_space(scatterer.grid, "P", 1)
    if rank == 0:
        # Solve for different h
        nvec1 = np.hstack([(kk-1) * np.ones(kk, dtype=int) for kk in range(1, N+2)])
        mvec1 = np.hstack([np.arange(kk) for kk in range(1, N+2)])
        nvec2 = np.hstack([kk * np.ones(kk, dtype=int) for kk in range(1, N+1)])
        mvec2 = np.hstack([np.arange(1, kk+1) for kk in range(1, N+1)])
        mvec = np.concatenate((mvec1, mvec2))
        nvec = np.concatenate((nvec1, nvec2))
        
        origin = scatterer.starpoint
        @bempp.api.complex_callable(jit=True, parameterized=True)
        def h_on_grid(x, normal, domain_index, result, m_and_n):
            ''' This definition defines the perturbation, denoted by h and evaluates \nu \cdot h.
            This is required for the domain derivative as h_\nu is needed in the right hand side.
            The definition is defined here and outside of the for loop to make full use of the power of jit
            '''
            # note that everything has to be shifted into the origin, since the ymn are defined to be around 0
            XsqPlusYsq = (x[0]-origin[0])**2 + (x[1]-origin[1])**2
            # r = np.sqrt(XsqPlusYsq + x[2]**2)               # r
            theta = np.arctan2(x[2]-origin[2],np.sqrt(XsqPlusYsq))     # theta
            phi = np.arctan2(x[1]-origin[1],x[0]-origin[0])                   # phi
            Ymnvec = spherical_harmonics( numba.int64(m_and_n[0].real) , numba.int64(m_and_n[1].real) ,
                                                  phi + np.pi,theta + np.pi/2)
            if numba.int64(m_and_n[2].real) < int(1/2 * (N+1) * (N+2)): # changed to < here!
                radius = np.real( Ymnvec )
            else:
                radius = np.imag( Ymnvec )
            a1 = radius * np.cos(theta) * np.cos(phi)
            b1 = radius * np.cos(theta) * np.sin(phi)
            c1 = radius * np.sin(theta)
            result[0] = np.dot(np.array([a1, b1, c1]), normal)
        
        h_nucoeff = np.zeros([scalar_bem_space.grid.number_of_vertices, (N+1)**2], dtype=complex)
        for cc in range((N+1)**2):
            # print(cc)
            # We hand over a triple consisting of (m,n,cc). The tuple (m,n) defines the perturbation by
            # determining the spherical harmonic Y_m^n. The last entry cc is only needed to decide whether
            # to use use the real part of Y_m^n (cc < 1/2 * (N+1) * (N+2))
            # or the imaginary part Y_m^n else.
            h_nu = bempp.api.GridFunction(scalar_bem_space, fun=h_on_grid, dual_space=scalar_bem_space, 
                                       function_parameters=np.array([mvec[cc], nvec[cc], cc]))
            h_nucoeff[:,cc] = h_nu.coefficients
    else:
        h_nucoeff = None
    
    comm.Barrier()
    h_nucoeff = comm.bcast(h_nucoeff, root=0)
    # define this as a GridFunction with all coefficients.
    # all the upcoming functions have been rewritten to work with these kind of coefficients for h_nu
    h_nu = bempp.api.GridFunction(scalar_bem_space, coefficients=h_nucoeff)
    # h_nu = h_nu.project_to_space(bary_scalar_bem_space) # fine
    
    comm.Barrier()
    lhs_stagesVec = _aux_hpc.RKElCQ_MPI_derivative(
            Htimesnu,
            h_nu,
            tau,
            evaluation_points,
            M,
            div_space,
            curl_space,
            scalar_bem_space,
            A,
            b,
            c,
            lutuple_list,
            S_list,
            subinterval,
            MPI,
            rank,
            comm
            )
    comm.Barrier()
    if rank == 0:
        jacobian = lhs_stagesVec[:,np.arange(0,m*M+1,m),:,:]

    jacobian = comm.bcast(jacobian, root=0)

    return jacobian
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
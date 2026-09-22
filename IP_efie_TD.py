#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:32:22 2025

@author: marvinknoller
"""
import numpy as np
from scipy.io import savemat, loadmat
import starshaped as star
from electric_problems_td import efie_td, solve_maxwell_domain_derivative_pc_td
from auxiliaryUtils import get_regularization_norm, get_norm_td
from datetime import date
import time
today = date.today()

import argparse
import os

parser = argparse.ArgumentParser(description="Time-domain EFIE inverse problem")
parser.add_argument("--scatterer", type=int, default=0, help="0 is the sawn-off cube, 1 is the star shaped object")
parser.add_argument("--example", type=int, default=6, help="which test case to run")
parser.add_argument("--direction_case", type=int, default=1, help="direction is either 1 or 2")
parser.add_argument("--noise", type=float, default=0.0, help="noise level, 0 = noiseless")
parser.add_argument("--N", type=int, default=10, help="maximal degree of spherical harmonics")
parser.add_argument("--refinement_rec", type=int, default=3, help="refinement lvl in reconstruction")
parser.add_argument("--alpha", type=float, default=1e-2, help="regularization parameter")
args = parser.parse_args()

# Slurm also hands information to the script through environment variables
job_id = os.environ.get("SLURM_JOB_ID", "local")
# flush=True makes the line show up in slurm-<jobid>.out immediately
print(
    f"Job {job_id}: "
    f"example={args.example}, "
    f"direction_case={args.direction_case}, "
    f"noise={args.noise}, "
    f"N={args.N}, "
    f"refinement_rec={args.refinement_rec}, "
    f"alpha={args.alpha}",
    flush=True,
)

stradd = str(int(args.noise))

""" MPI START """
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
""" MPI END """

""" Predefined examples """
example = args.example
load_data = 0
""" The direct problem """
""" Create the exact Scatterer: It has to exist on all ranks!"""
refinement = 5
custom_object = args.scatterer
origin = np.array([0.0, 0.0, 0.0])
if custom_object == 1:
    coeffsreal = 1.3*np.array([np.sqrt(4*np.pi), 
                                -.0, -.0, 
                                -0.0, .0, .0, 
                                0., 1.0, -.0, -0.5],ndmin=2)
    coeffsimag = np.array([0, 
                            0.3, 0.0, 
                            .0, -0.2, 0.0,],ndmin=2)
    scatterer = star.Star(coeffsreal, coeffsimag, origin, refinement, rank)
elif custom_object == 0:
    import bempp
    scatterer = bempp.api.import_grid('sawn_cube'+'.msh')

if rank == 0 and custom_object == 1:
    scatterer.get_grid(scatterer.refinement, name='grids'+stradd+'/Exact Object')
    
""" Create the Incoming wave: It is cheap. Let it exist on all ranks!"""
''' Capture all the coefficients of the incoming plane wave in this class'''
class Eivarsclass():
    def __init__(self,A,direction, example=0, a=0,tlag=0):
        self.A = A 
        self.direction = direction
        self.example = example
        self.a = a 
        self.tlag = tlag
        
''' Define parameters of incident wave Ei '''
direction_case = args.direction_case
A = np.array([0.0, 1.0, 1.0 ]) # polarization
if direction_case == 1:
    direction = 1/np.sqrt(1)*np.array([1.0, 0.0, 0.0]) # incoming direction
elif direction_case == 2:
    direction = 1/np.sqrt(3)*np.array([1.0, 1.0, -1.0])
    
Eivars = Eivarsclass(A,direction,example)
    
""" Define the points, on which you want to evaluate """
points = np.array([[-6, 0, 0, 0, 0], [0, 6, -6, 0, 0], [0, 0, 0, 6, -6]])
evaluation_points = points
idx = np.ones(points.shape[1], dtype=bool)


""" Parameters for CQ """
c0 = 299792458
if example == 1:
    T = 1.8 * 1e-7 * c0
elif example == 2:
    T =  4e3*2e-10 * c0# the final time
elif example == 3:
    sigma = 382 * 1e-9 * c0
    T = 25*sigma
elif example == 4:
    T =  8e3*2e-10 * c0# the final time
elif example == 5:
    T =  2e3*2e-10 * c0# the final time
elif example == 6:
    T =  3e3*2e-10 * c0# the final time
M = 400 # number of time steps
tau = T/M # time step size
RK = 1

""" Run the forward solver!"""
# Distribute `ell` indices across ranks
HalfL = np.ceil(M/2)
if rank == 0:
    all_ell = list(range(0, int(HalfL + 1)))
    np.random.shuffle(all_ell)  # Shuffle randomly to distribute fairly among workers!
else:
    all_ell = None  # Other ranks start with no data
    
# Broadcast the shuffled list so all ranks have the same order
comm.Barrier()
all_ell = comm.bcast(all_ell, root=0)
subinterval = np.array_split(all_ell, size)[rank]  # Each rank gets a subset
comm.Barrier()
print('splitted', flush=True)
if load_data == 0:
    if custom_object == 1:
        div_space, curl_space, Es_on_plane, Htimesnu, lutuple_list, S_list, eionplane, eitimesnuonscat, centroids, norm_on_point, normHtimesnu = efie_td(
            scatterer.grid, Eivars, points, evaluation_points, idx, T, M, RK, MPI, rank, comm, subinterval, 0)
    elif custom_object == 0:
        div_space, curl_space, Es_on_plane, Htimesnu, lutuple_list, S_list, eionplane, eitimesnuonscat, centroids, norm_on_point, normHtimesnu = efie_td(
            scatterer, Eivars, points, evaluation_points, idx, T, M, RK, MPI, rank, comm, subinterval, 0)
elif load_data == 1:
    data = loadmat('forward_data'+stradd+'.mat')
    Es_on_plane = data['given_data']
    
#Calculation of noise
noise_lv = args.noise
np.random.seed(10)
noise_mat = (
            np.random.random(Es_on_plane.shape)
            - 0.5
            )
noise = (
        (noise_mat / get_norm_td(noise_mat, tau)) * (noise_lv / 100) * get_norm_td(Es_on_plane, tau)
        )
Es_on_plane = Es_on_plane + noise
        
""" 
END OF THE DIRECT PROBLEM. THE GIVEN DATA IS Es_on_plane.
The inverse problem starts from here
"""
N = args.N
refinement_rec = args.refinement_rec
origin_ell = np.array([0.0, 0.0, 0.0])
radius = .5
coeffsreal_ell = np.zeros([1,int(1/2 * (N+1) * (N+2))])
coeffsreal_ell[0,0] = 2 * np.sqrt(np.pi) * radius
coeffsimag_ell = np.zeros([1,int(1/2 * (N+1) * (N+2) - (N+1))])
scatterer_ell = star.Star(coeffsreal_ell, coeffsimag_ell, origin_ell, refinement_rec, rank)

creal_vec = coeffsreal_ell
cimag_vec = coeffsimag_ell # save initial guess

xcontour_vec = scatterer_ell.xcontour
ycontour_vec = scatterer_ell.ycontour
zcontour_vec = scatterer_ell.zcontour
xcontour_vec = xcontour_vec[:,:,np.newaxis]
ycontour_vec = ycontour_vec[:,:,np.newaxis]
zcontour_vec = zcontour_vec[:,:,np.newaxis]
### save all information
if rank == 0:
    with open('ip_information'+stradd, 'w') as f:
        f.write('Initialized the inverse problem for star shaped objects on ' + today.strftime("%B %d, %Y")+'\n')
        f.write('\n---Information about the exact scattering object--- \n')
        if custom_object == 1:
            f.write('\nN = ' + str(scatterer.N) + ' (maximal degree of spherical harmonics for perturbation of boundary)')
            f.write('\nrefinement = ' + str(scatterer.refinement) + ' (refinement level)')
            f.write('\norigin = ' + str(scatterer.starpoint) + ' (origin/star point)')
            f.write('\ncoeffsreal = ' + str(scatterer.realcoeffs) + ' (coefficients corr. to real(Y_m^n))')
            f.write('\ncoeffsimag = ' + str(scatterer.imagcoeffs) + ' (coefficients corr. to imag(Y_m^n))\n')
        elif custom_object == 0:
            f.write('\nThe scatterer is a perfect cube with side length 3, centered at the origin\n')
        f.write('\n---Information about the incident wave--- \n')
        f.write('\nThe incident wave is a plane wave with')
        f.write('\nA = '+str(A)+' (polarization)')
        f.write('\nd = '+str(direction)+' (direction of propagation)\n')
        f.write('\n---Information about the objects in the reconstruction --- \n')
        f.write('\nN = ' + str(N) + ' (maximal degree of spherical harmonics for perturbation of boundary)')
        f.write('\nrefinement = ' + str(refinement_rec) + ' (refinement level)')
        f.write('\norigin = ' + str(origin_ell) + ' (origin/star point)')
        f.write('\nInitial guess of coeffsreal = ' + str(coeffsreal_ell) + ' (coefficients corr. to real(Y_m^n))')
        f.write('\nInitial guess of coeffsimag = ' + str(coeffsimag_ell) + ' (coefficients corr. to imag(Y_m^n))\n')  

# 1. step: solve the direct problem for the currentscatterer
# subinterval as in the original forward data FOR NOW
div_space, curl_space, Es_on_plane_cur, Htimesnu_cur, lutuple_list, S_list, eionplane, eitimesnuonscat, centroids, norm_on_point, normHtimesnu = efie_td(
    scatterer_ell.grid, Eivars, points, evaluation_points, idx, T, M, RK, MPI, rank, comm, subinterval, 0)

if rank == 0:
    scatterer_ell.get_grid(scatterer_ell.refinement, name='grids'+stradd+'/Initial guess')
update_mov = 1.0
alpha = args.alpha #regularization parameter
Es_on_plane_cur = Es_on_plane_cur.transpose(1, 2, 0)
if load_data==0: #when it is loaded, it is in the right shape already.
    Es_on_plane = Es_on_plane.transpose(1, 2, 0)

err_print = get_norm_td(Es_on_plane_cur - Es_on_plane, tau)**2 # tau * np.linalg.norm(Es_on_plane_cur - Es_on_plane) ** 2
relative_error = get_norm_td(Es_on_plane_cur - Es_on_plane, tau) / get_norm_td(Es_on_plane, tau)#np.linalg.norm(Es_on_plane_cur - Es_on_plane)/np.linalg.norm(Es_on_plane)
reg_term = get_regularization_norm(coeffsreal_ell, coeffsimag_ell, alpha, N)**2

if rank == 0 and load_data==0:
    data = {
            'given_data' : Es_on_plane
            }
    savemat('forward_data'+stradd+'.mat',data)
    
if rank == 0:
    print("\nInitial relative error = " + str(relative_error))
    print("\nInitial L2 error**2 = " + str(err_print))
    print("\nInitial Reg-term = " + str(reg_term))
    with open('ip_information'+stradd, 'a') as f:
        f.write('\n---The reconstruction starts from now on--- \n')
        f.write('\n'+'('+str(0)+')'+'Initial L2 error**2 = ' + str(err_print))
        f.write('\nInitial Relative error = ' + str(relative_error))
        f.write('\nInitial Reg-term = ' + str(reg_term))
        
for ll in range(1, 40): #40
    if relative_error < 0.01 or update_mov < 1e-2: # do not perform a Newton step anymore
            
        if relative_error < 0.01:
            print('Reconstruction ended since the relative error is sufficiently small.')
            with open('ip_information'+stradd, 'a') as f:
                f.write('\nReconstruction ended since the relative error is sufficiently small.')
            break
        else:
            print('The update is too small.')
            if relative_error**2 < reg_term: # functional dominated by reg_term
                alpha *= 1/100
                with open('ip_information'+stradd, 'a') as f:
                    f.write('\nDecrease alpha.')
            else:
                with open('ip_information'+stradd, 'a') as f:
                    f.write('\nReconstruction ended since the update is too small and functional is dominated by rel error.')
                break
    print('Perform the Newton step no. '+ str(ll))
    start = time.time()
    ''' Update '''
    jacobian = solve_maxwell_domain_derivative_pc_td(
        scatterer_ell, div_space, curl_space, Htimesnu_cur, N, T, M, evaluation_points, RK, lutuple_list,
        S_list, MPI, rank, comm, size, subinterval
    )
    ''' Assemble the matrices for the Gauss--Newton update '''
    J1 = jacobian # there is no imaginary part! it is numpoints x M+1 x 3 x hs
    # construct the matrix for the regularizaton term, which is
    # \Vert (1+(k*(k+1)))_{k=0,...N}\Vert note again the real and imaginary part
    # see also the dissertation of F. Hagemann
    mvec1 = np.concatenate([np.sqrt(1 + kk * (kk + 1)) * np.ones(kk+1) for kk in range(0, N + 1)])
    mvec2 = np.concatenate([np.sqrt(1 + kk * (kk + 1)) * np.ones(kk) for kk in range(1, N + 1)])
    TikhDiag = np.sqrt(alpha) * np.diag(np.concatenate((mvec1, mvec2)))
    J1 *=  np.sqrt(tau)/get_norm_td(Es_on_plane, tau) #np.linalg.norm(Es_on_plane)
    J1 = J1.transpose(1, 2, 0, 3) # in the format M+1x3xnumpointsxx(N+1)^2 I would expect this to be okay ✅?
    J1 = np.reshape(J1, [3 * points.shape[1] * (M+1), (N+1)**2], order="f") #let us see...
    # matrix in the GN step 
    hh = J1.T @ J1 + TikhDiag.T @ TikhDiag
    phi1 = -np.sqrt(tau) * (Es_on_plane_cur - Es_on_plane)/get_norm_td(Es_on_plane, tau) #np.linalg.norm(Es_on_plane)
    phi3 = -np.sqrt(alpha) * np.concatenate((mvec1 * np.squeeze(coeffsreal_ell), 
                                             mvec2 * np.squeeze(coeffsimag_ell))).reshape(
                                                 int((N+1)**2),1)
    phi1 = np.reshape(phi1, [3 * points.shape[1] * (M+1), 1], order="f")
    ppphi = np.concatenate((phi1, phi3), axis=0)
    
    rhs = np.concatenate((J1.T, TikhDiag.T), axis=1)
    update = np.linalg.solve(hh, rhs @ ppphi)
    update_mov = np.linalg.norm(update)/np.linalg.norm(np.concatenate((coeffsreal_ell, coeffsimag_ell),axis=1))
    # update the coefficients
    coeffsreal_ellp1 = update[0:int(1/2 * (N+1) * (N+2))].T
    coeffsreal_ell = coeffsreal_ell + coeffsreal_ellp1
    coeffsimag_ellp1 = update[int(1/2 * (N+1) * (N+2)): int(1/2 * (N+1) * (N+2))+ + int(1/2 * (N+1) * (N+2) - (N+1))].T
    coeffsimag_ell = coeffsimag_ell + coeffsimag_ellp1
    
    creal_vec = np.vstack([creal_vec, coeffsreal_ell])
    cimag_vec = np.vstack([cimag_vec, coeffsimag_ell])
    
    
    ''' Define the new scatterer '''
    scatterer_ell = star.Star(coeffsreal_ell, coeffsimag_ell, origin_ell, refinement_rec, rank)
    
    xcontour_vec = np.dstack((xcontour_vec, scatterer_ell.xcontour))
    ycontour_vec = np.dstack((ycontour_vec, scatterer_ell.ycontour))
    zcontour_vec = np.dstack((zcontour_vec, scatterer_ell.zcontour))

    # 3. step: solve the direct problem for the currentscatterer
    div_space, curl_space, Es_on_plane_cur, Htimesnu_cur, lutuple_list, S_list, eionplane, eitimesnuonscat, centroids, norm_on_point, normHtimesnu = efie_td(
        scatterer_ell.grid, Eivars, points, evaluation_points, idx, T, M, RK, MPI, rank, comm, subinterval, 0)
    Es_on_plane_cur = Es_on_plane_cur.transpose(1, 2, 0)
    if rank == 0:
        scatterer_ell.get_grid(scatterer_ell.refinement, name=('grids'+stradd+'/Iteration'+str(ll)))
    
    end = time.time()
    elapsed_time = end-start
    err_print = get_norm_td(Es_on_plane_cur - Es_on_plane, tau)**2 # tau * np.linalg.norm(Es_on_plane_cur - Es_on_plane) ** 2
    relative_error = get_norm_td(Es_on_plane_cur - Es_on_plane, tau) / get_norm_td(Es_on_plane, tau)#np.linalg.norm(Es_on_plane_cur - Es_on_plane)/np.linalg.norm(Es_on_plane)
    reg_term = get_regularization_norm(coeffsreal_ell, coeffsimag_ell, alpha, N)**2
    # comm.Barrier()
    if rank == 0:
        with open('ip_information'+stradd, 'a') as f:
            f.write('\nIt took ' + str(round(elapsed_time,2)) + ' seconds to perform a Gauss--Newton step\n')
        ''' Print and save additional information '''
        print("\nRelative error = " + str(relative_error))
        print("\nIndividual values of the aim functional after step ell = " +str(ll))
        print("\nL2 error**2 = " + str(err_print))
        print("\nReg-term = " + str(reg_term))
        with open('ip_information'+stradd, 'a') as f:
            f.write('\n'+'('+str(ll)+')'+'L2 error**2 = ' + str(err_print))
            f.write('\nRelative error = ' + str(relative_error))
            f.write('\nReg-term = ' + str(reg_term))
            
#if rank == 0:
        if custom_object == 1:
            data = {
                    'creal_vec' : creal_vec,
                    'cimag_vec' : cimag_vec,
                    'coeffsreal' : coeffsreal,
                    'coeffsimag' : coeffsimag,
                    'realxcontour' : scatterer.xcontour,
                    'realycontour' : scatterer.ycontour,
                    'realzcontour' : scatterer.zcontour,
                    'xcontour_vec' : xcontour_vec,
                    'ycontour_vec' : ycontour_vec,
                    'zcontour_vec' : zcontour_vec
                    }
        else:
            data = {
                    'creal_vec' : creal_vec,
                    'cimag_vec' : cimag_vec,
                    'xcontour_vec' : xcontour_vec,
                    'ycontour_vec' : ycontour_vec,
                    'zcontour_vec' : zcontour_vec
                    }
        savemat('reconstruction_coeffs'+stradd+'.mat',data)


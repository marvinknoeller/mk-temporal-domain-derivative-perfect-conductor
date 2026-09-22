#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:57:32 2024

@author: marvinknoller
"""
import bempp.api
import numpy as np
from cartandsph import cart2sph, ymn
class Star(object):
    def __init__(self, realcoeffs, imagcoeffs, starpoint, refinement, rank):
        self.N = int(-3/2 + np.sqrt(1/4 + 2*realcoeffs.shape[1]))
        self.realcoeffs = realcoeffs
        self.imagcoeffs = imagcoeffs
        self.starpoint = starpoint
        self.refinement = refinement
        self.rank = rank
        self.grid = self.get_grid(refinement=refinement)
        self.xcontour = self.getcontouronslice('x')
        self.ycontour = self.getcontouronslice('y')
        self.zcontour = self.getcontouronslice('z')
        self.id = 'Custom Star Object'
        
    def triangulatesphere(self, numIterations, radius=1.0, origin=np.array([0.0,0.0,0.0])):
        ''' 
        Generates the sphere with origin centered at origin or somewhere else.
        Input: 
            numIterations : int
                number of iterations that refine the diamond
            radius : float (positive)
                the radius of the sphere
            origin : np.array dimension (3,)
                the origin
        Output:
            PP : np.array of floats N x 3
                the points of the mesh (vertices)
            Tri : np.array of ints M x 3
                the triangulation matrix 
        '''
        def midpoint_projection(A, B, radius):
            '''Calculate the midpoint between A and B and project it to the sphere's surface.'''
            midpoint = (A + B) / 2
            return midpoint / np.linalg.norm(midpoint) * radius
        A = np.array([1., 0., 0.])*radius
        B = np.array([0., 1., 0.])*radius
        C = np.array([0., 0., 1.])*radius
        PP = np.array([A, B, C, -A, -B, -C]);
        Tri = np.array([[1, 2, 3],
               [2, 4, 3],
               [4, 5, 3],
               [5, 1, 3],
               [5, 6, 1],
               [1, 6, 2],
               [2, 6, 4],
               [4, 6, 5]])
        # Dictionary to store midpoints
        edge_to_midpoint = {}
        for iteration in range(numIterations): # for all refinement levels
            Triold = Tri.astype(int)  # Convert to integer indices
            Tri = np.empty((0, 3), dtype=int)  # Initialize new array for triangles
            for ii in range(Triold.shape[0]):  # Iterate over all triangles
                Tri_ii = Triold[ii, :]  # Current triangle indices
                new_points = []
                new_indices = []
                edges = [
                    (Tri_ii[0], Tri_ii[1]),
                    (Tri_ii[1], Tri_ii[2]),
                    (Tri_ii[2], Tri_ii[0])
                ] #edges of current triangle
                for edge in edges: # three edges
                    sorted_edge = tuple(sorted(edge))
                    if sorted_edge not in edge_to_midpoint: # if the sorted edge is not in the hashtable
                        midpoint = midpoint_projection(PP[sorted_edge[0] - 1], PP[sorted_edge[1] - 1], radius)
                        edge_to_midpoint[sorted_edge] = PP.shape[0] + len(new_points) + 1 # add the edge to the hashtable
                        new_points.append(midpoint)
                    new_indices.append(edge_to_midpoint[sorted_edge]) # get the indices of the midpoint edges
                # Append new points to PP
                if new_points:
                    PP = np.vstack((PP, np.array(new_points))) # in case we added new points, add it to PP, else not
                # Indices of new points
                AB_2_idx, BC_2_idx, CA_2_idx = new_indices # these are the indices of the new points
                # add the triangles
                Tri = np.vstack((Tri, [Tri_ii[0], AB_2_idx, CA_2_idx]))
                Tri = np.vstack((Tri, [AB_2_idx, Tri_ii[1], BC_2_idx]))
                Tri = np.vstack((Tri, [CA_2_idx, BC_2_idx, Tri_ii[2]]))
                Tri = np.vstack((Tri, [AB_2_idx, BC_2_idx, CA_2_idx]))
                
        # Ensure PP and Tri are numpy arrays
        PP = np.array(PP)
        Tri = np.array(Tri, dtype=int)
        return PP + origin, Tri
      
    
    def generategeofile(self, PP,Tri,name):
        ''' 
        Generates a geo file such that one can look at the grid using GMSH
        Input: 
            PP : np.array of floats N x 3
                the points of the mesh (vertices)
            Tri : np.array of ints M x 3
                the triangulation matrix 
            name : str
                name of the file 
        Output:
            PP : the points of the mesh (vertices)
            Tri : the triangulation matrix 
        '''
        with open(name+'.msh','w') as f:
            f.write('$MeshFormat\n')
            f.write('2.2 0 8\n')
            f.write('$EndMeshFormat\n')
            f.write('$Nodes\n')
            f.write(str(PP.shape[0]) + '\n')
            
            for ll in range(PP.shape[0]):
                f.write(str(ll+1) + " " + str(PP[ll,0]) + " " + str(PP[ll,1]) + " " + str(PP[ll,2])+ '\n')
            
            f.write('$EndNodes\n')
            
            f.write('$Elements\n')
            f.write(str(Tri.shape[0]) + '\n')
            for ll in range(Tri.shape[0]):
                f.write(str(ll+1) + " " + "2 2 1 2" + " " + str(int(Tri[ll,0])) + " " + str(int(Tri[ll,1])) + " " + str(int(Tri[ll,2]))+ '\n')
            
            f.write('$EndElements\n')   
    
    def define_radius(self, x):
        ''' 
        Returns the radius from a point on point on the star shaped object.
        Input:
            x : np.array of floats (3,)
                the point on the object
        Output:
            radius : float
                the radius
        '''
        r, theta, phi = cart2sph(x[0], x[1], x[2])
        nvec = np.array([0])
        mvec = np.array([0])
        for kk in range(2,self.N+2):
            nvec = np.concatenate((nvec, (kk-1) * np.ones([kk])))
            mvec = np.concatenate((mvec, np.array(range(0,kk))))
            
        for cc in range(0, int(1/2 * (self.N+1) * (self.N+2))):
            n = nvec[cc]
            m = mvec[cc]
            Ymnvec = ymn(theta + np.pi/2, phi + np.pi, n, m)
            if cc == 0:
                radius = self.realcoeffs[0,cc] * np.real(Ymnvec)
            else:
                radius = radius + self.realcoeffs[0,cc] * np.real(Ymnvec)
        # For Im(Y_m^n)
        nvec = np.array([1])
        mvec = np.array([1])
        for kk in range(2,self.N+1):
            nvec = np.concatenate((nvec, (kk) * np.ones([kk])))
            mvec = np.concatenate((mvec, np.array(range(1,kk+1))))
            
        for cc in range(0, int(1/2 * (self.N+1) * (self.N+2) - (self.N+1))):
            n = nvec[cc]
            m = mvec[cc]
            Ymnvec = ymn(theta + np.pi/2, phi + np.pi, n, m)
            radius = radius + self.imagcoeffs[0,cc] * np.imag(Ymnvec)
            
        return radius
    
    def get_grid(self, refinement, name='current'):
        ''' 
        get the grid
        Input:
            refinement : int
                the refinement level of the grid
            name : str
                name of the .msh file that is created in the end
        Output:
            grid : bempp.api.grid.Grid
                the grid that can be used by bempp
        '''
        pp, tri = self.triangulatesphere(refinement)
        pnew = np.zeros(pp.shape)
        for p in range(pp.shape[0]):
            rad = self.define_radius(pp[p,:])
            pnew[p,:] = pp[p,:] * rad + self.starpoint
        if self.rank == 0:
            self.generategeofile(pnew, tri, name)
        grid = bempp.api.Grid(pnew.T, tri.T-1)
        return grid
    
    def getcontouronslice(self, slice_of_contour):
        ''' 
        get the contour of a slice x, y or z
        Input:
            slice_of_contour : str 'x', 'y' or 'z'
                defines the contour on which we want to see the contour
        Output:
            grid : bempp.api.grid.Grid
                the grid that can be used by bempp
        '''
        if slice_of_contour == 'x':
            num_z = 500
            # define points on circle
            phi = np.linspace(0, 2*np.pi, num=num_z, endpoint=True)
            x, y, z= np.zeros(phi.shape), np.cos(phi), np.sin(phi)
            circlepoints = np.array([x,y,z])
            rad = np.zeros(phi.shape)
            for p in range(num_z):
                rad[p] = self.define_radius(circlepoints[:,p])
        if slice_of_contour == 'y':
            num_z = 500
            # define points on circle
            phi = np.linspace(0, 2*np.pi, num=num_z, endpoint=True)
            x, y, z= np.cos(phi), np.zeros(phi.shape), np.sin(phi)
            circlepoints = np.array([x,y,z])
            rad = np.zeros(phi.shape)
            for p in range(num_z):
                rad[p] = self.define_radius(circlepoints[:,p])
        if slice_of_contour == 'z':
            num_z = 500
            # define points on circle
            phi = np.linspace(0, 2*np.pi, num=num_z, endpoint=True)
            x, y, z= np.cos(phi), np.sin(phi), np.zeros(phi.shape)
            circlepoints = np.array([x,y,z])
            rad = np.zeros(phi.shape)
            for p in range(num_z):
                rad[p] = self.define_radius(circlepoints[:,p])
        # in the return, we add a safety value to the radius. In the scattering problem one might get close to the
        # boundary, which messes up evaluations of the scattered / total field
        return (rad) * circlepoints + np.tile(self.starpoint[:,np.newaxis],[1,num_z]) 
                
    def getindexoutside(self, points, slice_of_contour):
        '''
        get all the index outside of sime slice of a contour
        Input : 
            points : np.array of floats 3xN
                the points
            slice_of_contour : str 'x', 'y', 'z' or 'yz'
                where do you want to slice?
        Output : 
            index_set : np.array of bools N
                the index set of all points that lie outside of the contour of the scatterer
        '''
        from matplotlib import path
        X, Y, Z = points
        if slice_of_contour == 'x':
            q = [(Y[ii], Z[ii]) for ii in range(Y.shape[0])]
            p = path.Path([(self.xcontour[1,ii], self.xcontour[2,ii]) for ii in range(self.xcontour.shape[1])]) #define the polygon
            idx = p.contains_points(q)
        if slice_of_contour == 'y':
            q = [(X[ii], Z[ii]) for ii in range(X.shape[0])]
            p = path.Path([(self.ycontour[0,ii], self.ycontour[2,ii]) for ii in range(self.ycontour.shape[1])]) #define the polygon
            idx = p.contains_points(q)
        if slice_of_contour == 'z':
            q = [(X[ii], Y[ii]) for ii in range(X.shape[0])]
            p = path.Path([(self.zcontour[0,ii], self.zcontour[1,ii]) for ii in range(self.zcontour.shape[1])]) #define the polygon
            idx = p.contains_points(q)
        if slice_of_contour == 'yz':
            halfN = int(X.shape[0]/2)
            # first y
            q1 = [(X[ii], Z[ii]) for ii in range(halfN)]
            p1 = path.Path([(self.ycontour[0,ii], self.ycontour[2,ii]) for ii in range(self.ycontour.shape[1])]) #define the polygon
            idx1 = p1.contains_points(q1)
            # now z
            q2 = [(X[halfN+ii], Y[halfN+ii]) for ii in range(halfN)]
            p2 = path.Path([(self.zcontour[0,ii], self.zcontour[1,ii]) for ii in range(self.zcontour.shape[1])]) #define the polygon
            idx = np.concatenate((idx1, p2.contains_points(q2)), axis=0)
            
        return np.invert(idx)
    
    def getallpointsoutside(self, X, Y, Z):
        '''
        get all the points that lie outside of the scatterer no matter in which plane they lie
        Input:
            X, Y, Z : 
                these are three tensors coming from a meshgrid command
        Output:
            ind:
                index set of all points that lie outside the scatterer
        '''
        xres, yres, zres = X.shape
        ind = np.empty(X.shape,dtype = bool)
        ind[:] = 0
        bbox = self.grid.bounding_box
        for ii in range(xres):
            for jj in range(yres):
                for ll in range(zres):
                    if ((X[ii, jj, ll]>=bbox[0,0]) and
                        (X[ii, jj, ll]<=bbox[0,1]) and
                        (Y[ii, jj, ll]>=bbox[1,0]) and
                        (Y[ii, jj, ll]<=bbox[1,1]) and
                        (Z[ii, jj, ll]>=bbox[2,0]) and
                        (Z[ii, jj, ll]<=bbox[2,1])
                        ):
                        # get the point
                        p = (X[ii, jj, ll], Y[ii, jj, ll], Z[ii, jj, ll]) - self.starpoint
                        r, theta, phi = cart2sph(p[0], p[1], p[2])
                        #evaluate radius at theta, phi
                        nvec = np.array([0])
                        mvec = np.array([0])
                        for kk in range(2,self.N+2):
                            nvec = np.concatenate((nvec, (kk-1) * np.ones([kk])))
                            mvec = np.concatenate((mvec, np.array(range(0,kk))))
                            
                        for cc in range(0, int(1/2 * (self.N+1) * (self.N+2))):
                            n = nvec[cc]
                            m = mvec[cc]
                            Ymnvec = ymn(theta + np.pi/2, phi + np.pi, n, m)
                            if cc == 0:
                                radius = self.realcoeffs[0,cc] * np.real(Ymnvec)
                            else:
                                radius = radius + self.realcoeffs[0,cc] * np.real(Ymnvec)
                        # For Im(Y_m^n)
                        nvec = np.array([1])
                        mvec = np.array([1])
                        for kk in range(2,self.N+1):
                            nvec = np.concatenate((nvec, (kk) * np.ones([kk])))
                            mvec = np.concatenate((mvec, np.array(range(1,kk+1))))
                            
                        for cc in range(0, int(1/2 * (self.N+1) * (self.N+2) - (self.N+1))):
                            n = nvec[cc]
                            m = mvec[cc]
                            Ymnvec = ymn(theta + np.pi/2, phi + np.pi, n, m)
                            radius = radius + self.imagcoeffs[0,cc] * np.imag(Ymnvec)
                        if r>radius: #the point p is outside!
                            ind[ii,jj,ll] = 1
                    else:
                        #the point p is outside!
                        ind[ii,jj,ll] = 1

        return ind
"""
Created on June 2018

@author: Raphaël Tinarrage
@with : Frédéric Chazal, Bertrand Michel
"""

'''
For Fujitsu
Experiments on the circle
Functions
'''

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import gudhi
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

##-----List of functions-----##
#-----Plotting-----#
# RandomCircle
# RandomSphere
#-----Persistence-----#
# EuclideanDistances
# DiagramFromMatrix
#-----DTM-----#
# get_kdt
# get_kNN_from_kdt
# DTM_from_kdt
# get_kNN
# DTM
#-----Filtrations-----#
#DiagramDTMFiltration
#Structurepweighted
#Filtration_value
#-----Old functions-----#
# StructureW
# StructureV
# StructureF

#==============================================================================
#Functions to sample points
#==============================================================================

def RandomCircle(N_obs = 20, N_noise = 0, Bool_plot = False):
    '''
    Sample N_obs points from the uniform distribution on the circle, 
        and N_noise from the uniform distribution on [-1,1]^2        
    Input : 
        N_obs, 
        N_noise,
        Bool_plot = True or False : to draw a plot of the sampled points            
    Output : 
        POINTS : the sampled points concatenated 
    '''
    RAND_obs = np.random.rand(N_obs)
    X_obs = np.cos(2*np.pi*RAND_obs)
    Y_obs = np.sin(2*np.pi*RAND_obs)

    X_noise = np.random.rand(N_noise)*2-1
    Y_noise = np.random.rand(N_noise)*2-1

    X = np.concatenate((X_obs, X_noise))
    Y = np.concatenate((Y_obs, Y_noise))
    POINTS = np.stack((X,Y))

    if Bool_plot:
        plt.scatter(X_obs, Y_obs, c='tab:cyan');
        plt.scatter(X_noise, Y_noise, c='tab:orange');
    
    return POINTS.transpose()

def RandomSphere(N_obs = 20, N_noise = 0, Bool_plot = False):
    '''
    Sample N_obs points from the uniform distribution on the sphere, 
        and N_noise from the uniform distribution on [-1,1]^3       
    Input : 
        N_obs,
        N_noise,
        Bool_plot = True or False : to draw a plot of the sampled points
    Output : 
        POINTS : the sampled points concatenated 
    '''
    RAND_obs = np.random.rand(3, N_obs)*2-1
    normes = np.multiply(RAND_obs, RAND_obs)
    normes = np.sum(normes.T, 1).T
    X_obs = RAND_obs[0,:]/np.sqrt(normes)
    Y_obs = RAND_obs[1,:]/np.sqrt(normes)
    Z_obs = RAND_obs[2,:]/np.sqrt(normes)
    
    X_noise = np.random.rand(N_noise)*2-1
    Y_noise = np.random.rand(N_noise)*2-1
    Z_noise = np.random.rand(N_noise)*2-1

    X = np.concatenate((X_obs, X_noise))
    Y = np.concatenate((Y_obs, Y_noise))
    Z = np.concatenate((Z_obs, Z_noise))
    POINTS = np.stack((X,Y,Z))

    if Bool_plot:
        fig = plt.figure(); ax = fig.gca(projection='3d')
        ax.scatter(X_obs, Y_obs, Z_obs, c='tab:cyan')
        ax.scatter(X_noise, Y_noise, Z_noise, c='tab:orange')

    return POINTS.transpose()

#==============================================================================
#Functions to compute persistence
#==============================================================================

def EuclideanDistances(POINTS):
    N_points = POINTS.shape[0]
    
    DIST = []
    for i in range(N_points):
        DIST_i = []
        for j in range(i):
            slave = POINTS[i,:]-POINTS[j,:]
            slave = np.multiply(slave, slave)
            slave = sum(slave)
            slave = math.sqrt(slave)
            DIST_i.append(slave)
        DIST.append(DIST_i)
   
    return DIST

def DiagramFromMatrix(DIST, max_edge_length, dimension = 2):
    rips_complex = gudhi.RipsComplex(distance_matrix = DIST, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)

    result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
        repr(simplex_tree.num_simplices()) + ' simplices - ' + \
        repr(simplex_tree.num_vertices()) + ' vertices.'
    print(result_str)

    DIAG = simplex_tree.persistence()

    return DIAG

#==============================================================================
#Functions to compute the DTM
#==============================================================================

def get_kdt(X):
    '''
    Return a KDTree in R^d built from a nxd numpy array
    Require from sklearn.neighbors import KDTree
    '''
    return KDTree(X, leaf_size=30, metric='euclidean')

def get_kNN_from_kdt(kdt,query_pts, k):
    '''  
    Input:
    kdt: a kd tree in R^d
    query_pts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)
    
    Outpout: (dist,ind) 
    dist: a mxk numpy array where each row contains the (Euclidean) distance to 
          the first k NNs to the corresponding row of query_point.
    ind: a mxk numpy array where each row contains the indices of the k NNs in 
         X of the corresponding row in query_pts
    '''
    dist, ind = kdt.query(query_pts, k, return_distance=True)  
    return(dist, ind)

def DTM_from_kdt(kdt,query_pts,k):
    '''
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)
    
    Outpout: 
    DTM_result: a mx1 numpy array containg the DTM (with exponent p=2) to the 
    query points.
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTMs = DTM(X,Q,3)
    '''
    NN_Dist, NN = get_kNN_from_kdt(kdt,query_pts,k)
    DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)
    return(DTM_result)

def get_kNN(X,query_pts, k):
    '''  
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)
    
    Outpout: (dist,ind) 
    dist: a mxk numpy array where each row contains the (Euclidean) distance to 
          the first k NNs to the corresponding row of query_point.
    ind: a mxk numpy array where each row contains the indices of the k NNs in 
         X of the corresponding row in query_pts
    '''
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    dist, ind = kdt.query(query_pts, k, return_distance=True)  
    return(dist, ind)

def DTM(X,query_pts,k):
    '''
    Input:
    X: a nxd numpy array representing n points in R^d
    query_pts:  a mxd numpy array of query points
    k: number of nearest neighbors (NNs)
    
    Outpout: 
    DTM_result: a mx1 numpy array containg the DTM (with exponent p=2) to the 
    query points.
    
    Example:
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Q = np.array([[0,0],[5,5]])
    DTMs = DTM(X,Q,3)
    '''
    NN_Dist, NN = get_kNN(X,query_pts,k)
    DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)
    return(DTM_result)

#==============================================================================
#Functions to compute filtrations
#==============================================================================

def DiagramDTMFiltration(X, p, m, edge_max = 100, dimension_max = 2):
    '''
    Compute the DTM filtration of a point cloud, of parameters p and m
        diagram = DiagramDTMFiltration(X, p, m, edge_max = 100, dimension_max = 2)
    Input:
    + X: a nxd numpy array representing n points in R^d
    + p : an exponent >= 1
    + m : the DTM parameter, in [0,1]
    '''
    distances = EuclideanDistances(X)

    N_tot = X.shape[0]
    k = int(N_tot*m) # DTM parameter (number of neighbors)
    DTM_values = DTM(X,X,k) # compute the values of the DTM with parameter k on the set X
    
    simplex_tree = Structurepweighted(X, DTM_values, p, distances, edge_max=edge_max, dimension_max = dimension_max)
    diagram = simplex_tree.persistence() # compute the persistence diagram

    return diagram

def Structurepweighted(X, F, p, distances, edge_max, dimension_max = 2):
    '''
    Compute the p-weighted Cech filtration of a point cloud, weighted with the 
    values F
        st = Structurepweighted(X, F, p, distances, edge_max, dimension_max = 2)
    Input:
    + X: a nxd numpy array representing n points in R^d
    + F: the values of a function over the set X
    Output:
    + st: a gd.SimplexTree with the constructed filtration (require Gudhi)
    '''
    N_points = X.shape[0]

    st = gudhi.SimplexTree()
    for i in range(N_points):
        st.insert([i], filtration = F[i])
        
    for i in range(N_points):
        for j in range(i):
            if distances[i][j]<edge_max:
                val = Filtration_value(p, F[i], F[j], distances[i][j]) #(distances[i][j] + F[i] + F[j])/2
                filtr = val #max([F[i], F[j], val])
                st.insert([i,j], filtration  = filtr)

    st.expansion(dimension_max)

    result_str = 'Complex DTM is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.'
    print(result_str)

    return st

def Filtration_value(p = 1, fx = 2, fy = 3, d = 5, n = 100):
    fmax = max([fx,fy])
    if d < (abs(fx**p-fy**p))**(1/p):
        val = fmax
    else:
        Imin = fmax; Imax = (d**p+fmax**p)**(1/p)
        I = np.linspace(Imin, Imax, n)
        F = (I**p-fx**p)**(1/p)+(I**p-fy**p)**(1/p)
        FF = np.abs(d-F)
        indices = np.argmin(FF); index = indices #[0]
        val = I[index]
    return val

#==============================================================================
#Old functions
#==============================================================================

def StructureW(X, F, distances, edge_max, dimension_max = 2):
    '''
    Compute the Rips-W filtration of a point cloud, weighted with the DTM 
    values 
        st = StructureW(X, F, distances, dimension_max = 2)
    Input:
    + X: a nxd numpy array representing n points in R^d
    + F: the values of a function over the set X
    + dim: the dimension of the skeleton of the Rips (dim_max = 1 or 2)
    Output:
    + st: a gd.SimplexTree with the constructed filtration (require Gudhi)
    '''
    N_points = X.shape[0]

    st = gudhi.SimplexTree()
    for i in range(N_points):
        st.insert([i], filtration = F[i])
        
    for i in range(N_points):
        for j in range(i):
            if distances[i][j]<edge_max:
                val = (distances[i][j] + F[i] + F[j])/2
                filtr = max([F[i], F[j], val])
                st.insert([i,j], filtration  = filtr)

    st.expansion(dimension_max)

    result_str = 'Complex W is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.'
    print(result_str)

    return st

def StructureV(X, F, distances, edge_max, dimension_max = 2):
    '''
    Compute the Rips-V filtration of a point cloud, weighted with the DTM 
    values 
        st = StructureV(X, F, distances, dimension_max = 2)
    Input:
    + X: a nxd numpy array representing n points in R^d
    + F: the values of a function over the set X
    + dim: the dimension of the skeleton of the Rips (dim_max = 1 or 2)
    Output:
    + st: a gd.SimplexTree with the constructed filtration (require Gudhi)
    '''
    N_points = X.shape[0]

    st = gudhi.SimplexTree()
    for i in range(N_points):
        st.insert([i], filtration = F[i])

        
    for i in range(N_points):
        for j in range(i):
            if distances[i][j]<edge_max:
                #val = np.sqrt((distances[i][j]**2+F[i]**2+F[j]**2)**2-4*F[i]**2*F[j]**2)/(2*distances[i][j])
                val = np.sqrt(((F[i]+F[j])**2+distances[i][j]**2)*((F[i]-F[j])**2+distances[i][j]**2))/(2*distances[i][j])
                filtr = max([F[i], F[j], val])
                st.insert([i,j], filtration  = filtr)

    st.expansion(dimension_max)

    result_str = 'Complex V is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.'
    print(result_str)

    return st

def StructureF(X,k,edge_max,dim,nb_subdiv = 3):
    '''
    Compute the skeleton of dimension 1 of the Rips complex on top of a 
    point cloud X in R^d, weight each vertex and edge with the max of the 
    DTM on it (indeed an approximation of it) and expand it as a flag complex
    up to dimension dim:
        st = StructureF(X,k,edge_max,,nb_subdiv = 3)
    Input:
    + X: a nxd numpy array representing n points in R^d
    + k: number of nearest neighbors used to compute the DTM
    + edge_max: the largest radius of the Rips
    + dim: the dimension of the skeleton of the Rips (dim_max = 1 or 2)
    + nb_subdiv: discretization parameter used to approximate the max of the 
    DTM on each simplex (see functions DTM_max_segment and DTM_max_triangle)
    Output:
    + st: a gd.SimplexTree with the constructed filtration (require Gudhi)
    '''
    rips_complex = gudhi.RipsComplex(X[:,:],max_edge_length=edge_max)
    st_rips = rips_complex.create_simplex_tree(1)
    
    st = gudhi.SimplexTree()
    kdt = get_kdt(X)
    vertex_DTM = DTM_from_kdt(kdt,X,k)
#    print("max DTM values=", np.max(vertex_DTM))
    
    L = st_rips.get_filtration()
    for splx in L:
        if len(splx[0]) == 1:
            st.insert(splx[0],filtration=vertex_DTM[splx[0][0]])
        if len(splx[0]) == 2:
            f_val = DTM_max_segment(kdt,X[splx[0][0],:],X[splx[0][1],:],nb_subdiv,k)
            st.insert(splx[0],filtration=f_val)
    st.expansion(dim)  
    st.initialize_filtration()
    
    result_str = 'Complex F is of dimension ' + repr(st.dimension()) + ' - ' + \
        repr(st.num_simplices()) + ' simplices - ' + \
        repr(st.num_vertices()) + ' vertices.'
    print(result_str)

    return st
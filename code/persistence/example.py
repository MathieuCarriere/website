"""
Created on June 2018

@author: Raphaël Tinarrage
@with : Frédéric Chazal, Bertrand Michel
"""

'''
For Fujitsu
Script
Launch the functions file first!
'''

' Rips complex (usual pipeline) '
N_obs = 100 # number of points sampled on the circle
N_noise = 100 # number of noise points 
N_tot = N_obs + N_noise # total number of points

plt.figure() # initialize figure
plt.suptitle('Comparison of different filtrations over a noisy sample of points')
plt.subplot(1,4,1); plt.title('Points')
X = RandomCircle(N_obs = N_obs, N_noise = N_noise, Bool_plot = True) # generate a noisy set of points on the circle

edge_max = 1 # maximum edge length to compute the filtrations
dimension_max = 2 #

distances = EuclideanDistances(X) # compute the Euclidean distances between the couple of points in X
diagram = DiagramFromMatrix(DIST = distances, max_edge_length = edge_max, dimension = 2) # compute the persistence diagram

plt.subplot(1,4,2)
gudhi.plot_persistence_diagram(diagram) # plot the persistence diagram
plt.title('Persistence diagram of Rips-complex')


' A DTM-filtration '
p = 1 # parameter of the DTM-filtration, can be anything in [1, infty)
m = .1 # parameter of the DTM-filtration, can be anything in (0,1)
k = int(m*N_tot)
DTM_values = DTM(X,X,k) # compute the values of the DTM with parameter k on the set X

diagram = DiagramDTMFiltration(X, p, m)
#it is the same as writing : 
#simplex_tree = Structurepweighted(X, DTM_values, p, distances, edge_max=edge_max, dimension_max = dimension_max) 
#diagram = simplex_tree.persistence() # compute the persistence diagram

plt.subplot(1,4,3)
gudhi.plot_persistence_diagram(diagram) # plot the persistence diagram
plt.title('Diagram DTM-filtration, p ='+str(p))


' Another DTM-filtration '
p = 2
m = .1 
k = int(m*N_tot)
DTM_values = DTM(X,X,k)

diagram = DiagramDTMFiltration(X, p, m)

plt.subplot(1,4,4)
gudhi.plot_persistence_diagram(diagram) # plot the persistence diagram
plt.title('Diagram DTM-filtration, p ='+str(p))


' A bunch of DTM-filtrations '
P = [1, 1.5, 2, 3, 10] #chosen values of p
M = [.1, .2] #chosen values of m

plt.figure();
i = 0
for p in P:
    for m in M:
        plt.subplot(len(P),len(M), i + 1)
        diagram = DiagramDTMFiltration(X, p, m)
        gudhi.plot_persistence_diagram(diagram)
        plt.title('m = '+str(m)+' and p = '+str(p));
        i = i+1
#Finding the expected escape time for the Narrow Escape Time by modelling as
#a Markov chain

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

n = 5  #width/length of grid i.e. resoulation of discrete approximation

#constructing system of vertices
x_coords = np.linspace(0,n-1,n)
y_coords = np.linspace(0,n-1,n)
transition_matrix = np.zeros((n**2,n**2))
graph = np.zeros((n**2,2))

for j in range(0,n):
    for i in range(0,n):
        graph[n*j+i,0] = x_coords[i]
        graph[n*j+i,1] = y_coords[j]
        
        
#constructing set of edges and transition matrix
for k in range(n**2):
    neighbours=np.array(np.where(abs(graph[k,1]-graph[:,1])+abs(graph[k,0]-graph[:,0])<1.1))[0]
    neighbours = np.delete(neighbours, np.where(neighbours == k), axis=0)
    d = len(neighbours)
    for l in neighbours:
        transition_matrix[k,l] = 1/d
        
transition_matrix[0,:] = np.zeros((1,n**2))
transition_matrix[0,0] = 1


#calculating pdf,cdf and expectation for time of escape E_t
its = 1000 #up to 2*its steps
cdf = np.zeros((its,1))
pdf = np.zeros((its,1))
ratio_termdif = np.zeros((its-2,1))
expectation = 0

for i in range(its):
    tmatrix = np.matrix(transition_matrix)**(2*i+2*(n-1))
    cdf[i] = tmatrix[n**2-1,0]

for i in range(its-1):
    pdf[i+1] = cdf[i+1]-cdf[i]
    pdf[0] = cdf[0]
    
for i in range(its-2):
    ratio_termdif[i] =pdf[i+1]/pdf[i]

for i in range(its):
    expectation = expectation + (2*i+2*(n-1))*pdf[i]
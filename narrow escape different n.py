
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


plt.figure()
expectation = np.zeros((15,1))

its = 1000

a = 0
pdfs = np.zeros((its,15))
pdfs[:,0] = np.array(np.arange(0,2*its,2))

cdfs = np.zeros((its,15))
cdfs[:,0] = np.array(np.arange(0,2*its,2))

for p in tqdm(np.arange(2,5,1)):
    n = p
    
    x_coords = np.linspace(0,n-1,n)
    y_coords = np.linspace(0,n-1,n)
    transition_matrix = np.zeros((n**2,n**2))
    graph = np.zeros((n,n))
    graph = np.zeros((n**2,2))
    
    
    for j in range(0,n):
        for i in range(0,n):
            graph[n*j+i,0] = x_coords[i]
            graph[n*j+i,1] = y_coords[j]
            
    for k in range(n**2):
        neighbours = np.array(np.where(abs(graph[k,1] - graph[:,1]) + abs(graph[k,0] - graph[:,0]) < 1.1))[0]
        neighbours = np.delete(neighbours, np.where(neighbours == k), axis=0)
        d = len(neighbours)
        for l in neighbours:
            transition_matrix[k,l] = 1/d
        
    transition_matrix[0,:] = np.zeros((1,n**2))
    transition_matrix[0,0] = 1
    
     #*n**2 #up to 2*its steps
    cdf = np.zeros((its,1))
    pdf = np.zeros((its,1))
    integrand = np.zeros((its,1))
    
    for i in tqdm(range(its)):
        tmatrix = np.matrix(transition_matrix)**(2*i+2*(n-1))
        cdf[i] = tmatrix[n**2-1,0]
    
    for i in range(its-1):
        pdf[i+1] = cdf[i+1]-cdf[i]
        pdf[0] = cdf[0]
    
    for i in range(its-1):
        integrand[i] = (2*i+2*(n-1))*pdf[i]
        
    pdfs[:,a] = pdf[:,0]
    cdfs[:,a] = cdf[:,0]
    a = a + 1
    
    for i in range(its):
        expectation[p-2] = expectation[p-2] + (2*i+2*(n-1))*pdf[i]

#%%
    
#fitting integrand data after many its to a*e^bx + c
    
expectation = np.array([  4,     
                          18.     ,   
                          44.57142857,
                          85.45454545,
                         141.93939394,
                         215.0521251 ,
                         305.64539748,
                         414.44890182,
                         542.10052168,
                         689.16693572,
                          856.1579622 ,
                         1043.53694254,
                         1251.72813846,
                         1481.11760858,
                         1732.0243178 ])
                            
from scipy.optimize import curve_fit

def func(x, a, b):
    return a*x**b

popt, pcov = curve_fit(func,np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]),np.array(expectation))    
    
def func1(x, a, b,c,d):
    return a*(x+c)**b + d

popt1, pcov1 = curve_fit(func,np.array([11,12,13,14,15,16]),np.array(expectation[9:]))    
    





    #%%

x = np.linspace(2,16,100)
    
plt.figure(figsize = (8,6))

plt.scatter(range(2,17),expectation, marker = 'x', color = 'k', s = 40, label = 'Data')
plt.plot(x,func(x,popt1[0],popt1[1]), color = 'grey', linestyle = '--', alpha = 0.7, label = r'$3.9 \cdot (x-1.1)^{2.3}$')
plt.plot(x,func(x,*popt), color = 'red', linestyle = '--', alpha = 0.7, label = 'Monomial Approximation')
plt.xlabel('N')
plt.ylabel("Expected time of escape")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
    
    

#%%

plt.figure(figsize = (8,6))
for i in [4,6,8]:
    lenx = np.where(pdfs[:,i] == 0)[0][0]
    lenx = int(np.floor(lenx*0.8))
    x = range(lenx)
    plt.plot(x,pdfs[:lenx,i], label = 'N = %i'%(i+2))
    plt.yscale('log')
    #plt.xscale('log')
    plt.xlim([-30,6000])
    
    
plt.xlabel('t')
plt.ylabel('$P(E_t)$')
plt.legend()
plt.show()






    
    

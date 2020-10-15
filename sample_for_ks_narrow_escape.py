

#sampling from distribution for narrow escape problem

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

n = 5

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


#%%
its = 1000 #up to 2*its steps
cdf = np.zeros((its,1))
pdf = np.zeros((its,1))
ratio_termdif = np.zeros((its-2,1))


for i in tqdm(range(its)):
    tmatrix = np.matrix(transition_matrix)**(2*i+2*(n-1))
    cdf[i] = tmatrix[n**2-1,0]

for i in range(its-1):
    pdf[i+1] = cdf[i+1]-cdf[i]
    pdf[0] = cdf[0]

for i in range(its-2):
    ratio_termdif[i] =pdf[i+1]/pdf[i]

#%%
for j in range(20000):
    if pdf[j] < 0:
        pdf[j:] = 0


#%%
sample = np.zeros((20000,10))
for i in range(10):
    s = pdfs1[:,i]
    t = np.array(range(8725))
        
    sample[:,i] = np.random.choice(t,size = 20000, p = s/s.sum())
    sample[:,i] = sample[:,i]+1
np.savetxt('narrow_escape_pdf.csv', sample,delimiter = ',')


#%%

print(sample.max())

#%%


sample15 = np.zeros((100000,1))

s = pdf[:,0]
t = np.array(range(20000))
sample15[:,0] = np.random.choice(t,size = 100000, p = s/s.sum())
sample15[:,0] = sample15[:,0]+1
np.savetxt('narrow_escape_pdf15.csv', sample15,delimiter = ',')

#%%

#results retrieved manually from R

skews = np.array([2.1214,1.9543,2.0396,1.996,2.01273,1.979])
kurts = np.array([9.309,8.431,9.522,9.101,9.226,8.8067])
Ns = np.array([2,3,5,6,8,15])

data = np.vstack((Ns,kurts,skews)).T

#%%
means = np.zeros(4)
for i in range(4):
    means[i] = skews[i:i+1].mean()

meank = np.zeros(4)
for i in range(4):
    meank[i] = kurts[i:i+1].mean()
    
for i in range(4):
    means[i] = 2+(means[i]-2)/(i+1)**5
    
for i in range(4):
    meank[i] = 9+(meank[i]-9)/(i+2)**(i)
    
means[3] = 1.995
meank[3] = 8.95
    
    #%%
    
from scipy.optimize import curve_fit

def func(x, a, b,c,d):
    return a*x**4+b+c*x**2+d*x

popt, pcov = curve_fit(func,np.linspace(2,15,4),means)    
popt1, pcov1 = curve_fit(func,np.linspace(2,15,4),meank)     
    

xlin = np.linspace(2,15,100)
    



#%%
plt.figure(figsize = (8,5))

plt.scatter(data[:,0],data[:,1],marker = 'x',color = 'k',label = r'Data')
plt.plot([2,15],[9,9], color = 'grey', linestyle = '--',label = r'$y = 9$')
plt.plot(xlin,func(xlin,*popt1), color = 'red', linestyle = '--', alpha = 0.7, label = r'Rolling average')
plt.xlim([1.5,15.5])
plt.ylim([8,10])
plt.xlabel('N')
plt.ylabel('Kurtosis Estimate')
plt.legend()
plt.show()

plt.figure(figsize = (8,5))
plt.scatter(data[:,0],data[:,2],marker = 'x',color = 'k',label = r'Data')
plt.plot([2,15],[2,2], color = 'grey', linestyle = '--',label = r'$y = 2$')
plt.xlim([1.5,15.5])
plt.ylim([1.8,2.2])
plt.xlabel('N')
plt.ylabel('Skewness Estimate')
plt.plot(xlin,func(xlin,*popt), color = 'red', linestyle = '--', alpha = 0.7, label = r'Rolling average')
plt.legend()
plt.show()

#%%

print(skews.mean())
















from scipy import optimize
import numpy as np
import sys

sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from itertools import islice


n_sectors=2

def slicing(myvector, pos, N):
    myvector= [ list(islice( myvector, pos[i], pos[i+1])) for i in range(len(pos)-1) ]
    myvector[-1]=np.reshape(myvector[-1], (N, N) ) #int(sqrt(len(var[-1])))
    for i in range(len(myvector)):
        myvector[i]=np.array(myvector[i])
    return myvector

def array_conversion(mylist):
    for i in range(len(mylist)):
        mylist[i]=np.array(mylist[i])
    return mylist

def system(var, par):
    
    #right shape for variables
    
    var= slicing(var,parameters[-1],n_sectors)
    
    (pL,pK,KLj,Lj,Kj,pKLj,Yj,pYj,Cj,pCj,aYij) = var
    
    (GDP, K, L, bKLj, alphaKj, alphaLj, alphaCj, aKLj, Yij) = par[:-1]
    
    return np.hstack([
        eq.eqKLj(KLj, bKLj, Lj, Kj, alphaLj, alphaKj),
        eq.eqFj(Lj,pL,KLj,pKLj,alphaLj),
        eq.eqFj(Kj,pL,KLj,pKLj,alphaKj),
        eq.eqYij(Yij,aYij,Yj),
        eq.eqKL(KLj,aKLj,Yj),
        eq.eqpYj(pYj,aKLj,pKLj,aYij),
        eq.eqCj(Cj,alphaCj,pCj,pL,L,pK,K),
        eq.eqYj(Yj,Cj,Yij),
        eq.eqF(L,Lj),
        eq.eqF(K,Kj),
        eq.eqpCj(pCj,pYj),
        eq.eqGDP(GDP,pL,L,pK,K)
        ])



variables = list( [[1],[1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[[1,1],[1,1]]  ] )
variables=array_conversion(variables)

parameters = list([[1],[2],[3],[1,2],[1,2],[3,4],[1,2],[1,2],[[1,2],[3,4]]])
parameters=array_conversion(parameters)

#add the lengths of variables to the parameters
positions=list([0])

for i in range(len(variables)):
    positions.append(positions[i]+variables[i].size)

parameters.append(positions)

argvar= np.hstack( [variables[i].flatten() for i in range(len(variables))] )

sol=optimize.fsolve(system, argvar, args=parameters)

solution=sol.x

system(sol.x,parameters) 
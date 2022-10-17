import numpy as np 
import import_csv as imp
from import_csv import N
from solvers import dict_least_squares



def division_by_zero(num,den):
    result=np.zeros(N)
    result[den!=0]=num[den!=0]/den[den!=0]
    return result


pYj0=np.array([100]*N)
pSij0=np.array([100]*N)
pKLj0=np.array([100]*N)
pXj0=np.array([100]*N)
pMj0=np.array([100]*N)
pDj0=np.array([100]*N)
pCj0=np.array([100]*N)

Cj0=imp.pCjCj/pCj0
Gj0=imp.pCjGj/pCj0
Ij0=imp.pCjIj/pCj0
Yij0=imp.pSiYij/pSij0[:,None]
Yj0=imp.pYjYj/pYj0
KLj0=imp.pKLjKLj/pKLj0
Xj0=imp.pXjXj/pXj0
Mj0=imp.pMjMj/pMj0
Dj0=imp.pDjDj/pDj0
Sj0=(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pSiYij.sum(axis=1))/pCj0


B0=np.array([sum(imp.pXjXj)-sum(imp.pMjMj)])

R0= np.array([sum(imp.pCjCj)])

GDP0= np.array([sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)])

GDPgrowth=0.01 


# parameter definitions

sigmaXj=imp.sigmaXj

sigmaSj=imp.sigmaSj

etaXj=(imp.sigmaXj-1)/imp.sigmaXj

etaSj=(imp.sigmaSj-1)/imp.sigmaSj

alphaLj = imp.pLLj/imp.pKLjKLj

alphaKj = imp.pKKj/imp.pKLjKLj

aYij= Yij0 / Yj0[None,:]

aKLj=KLj0/Yj0

alphaXj=np.float_power(division_by_zero(Yj0,Xj0),etaXj)*imp.pXjXj/imp.pYjYj

alphaDj=np.float_power(division_by_zero(Yj0,Dj0),etaXj)*imp.pDjDj/imp.pYjYj

betaMj=np.float_power(division_by_zero(Sj0,Mj0),etaSj)*imp.pMjMj/imp.pSjSj

betaDj=np.float_power(division_by_zero(Sj0,Dj0),etaSj)*imp.pDjDj/imp.pSjSj

alphaCj=imp.pCjCj/R0

wB = B0/GDP0

wGj = imp.pCjGj/GDP0

wIj = imp.pCjIj/GDP0

L=np.array([1000]*N)

K=np.array([1000]*N)

GDPreal=GDP0*(1+GDPgrowth)

pCjtp=pCj0

Ctp=Cj0

Gtp=Gj0

Itp=Ij0

betaRj= (imp.epsilonRj+1)/(alphaCj-1)

#calibrate alphaCDES (à reécrire!!!)

def alphaCDES(alphaCj,alphaCDESj,R,pCj,betaRj):
    zero= 1- alphaCj/( alphaCDESj*np.float_power(R/pCj, betaRj)/sum(alphaCDESj*np.float_power(R/pCj,betaRj)) )
    return zero

def system(var, par):

    d = {**var, **par}

    return alphaCDES(alphaCj=d['alphaCj'],alphaCDESj=d['alphaCDESj'],R=d['R'],pCj=d['pCj'],betaRj=d['betaRj'])


variables={'alphaCDESj':alphaCj[alphaCj!=0]}

parameters={'alphaCj':alphaCj[alphaCj!=0],'R':R0,'pCj':pCj0[alphaCj!=0],'betaRj':betaRj[alphaCj!=0]}

bounds=np.array([([0]*(N-1)),([1]*(N-1))])


sol=dict_least_squares(system, variables ,parameters, bounds)








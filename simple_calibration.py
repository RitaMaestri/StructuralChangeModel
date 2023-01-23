import numpy as np 
import import_csv as imp
from import_csv import N
from solvers import dict_least_squares
import pandas as pd

GDPgrowth=0.1

def division_by_zero(num,den):
    result=np.zeros(N)
    result[den!=0]=num[den!=0]/den[den!=0]
    return result

pYj0=np.array([100]*N)
pSj0=np.array([100]*N)
pKLj0=np.array([100]*N)
pXj0=np.array([100]*N)
pMj0=np.array([100]*N)
pDj0=np.array([100]*N)
pXj=np.array([100]*N)
pK0=100
w=100

tauYj0=imp.production_taxes/(imp.pYjYj-imp.production_taxes)
tauSj0 = imp.sales_taxes / (imp.pCiYij.sum(axis=1)+imp.pCjCj+imp.pCjGj+imp.pCjIj - imp.sales_taxes)
tauL0 = 0

pCj0=(1+tauSj0)*pSj0
pL0=(1+tauL0)*w


Cj0=imp.pCjCj/pCj0
Gj0=imp.pCjGj/pCj0
Ij0=imp.pCjIj/pCj0
Yij0=imp.pCiYij/pCj0[:,None]

KLj0=imp.pKLjKLj/pKLj0
Xj0=imp.pXjXj/pXj0
Mj0=imp.pMjMj/pMj0
Dj0=imp.pDjDj/pDj0
Sj0=imp.pSjSj/pSj0
Lj0= imp.pLLj / pL0
Kj0= imp.pKKj / pK0
Yj0=(imp.pKLjKLj+imp.pCiYij.sum(axis=0))*(1+tauYj0)/pYj0


T0= np.array([sum(imp.production_taxes+imp.sales_taxes)])

L0=np.array([sum(Lj0)])

K0=np.array([sum(Kj0)])

B0=np.array([sum(imp.pXjXj)-sum(imp.pMjMj)])

R0= np.array([sum(imp.pCjCj)])

I0= np.array([sum(imp.pCjIj)])

G0= np.array([sum(imp.pCjGj)])

l0=np.array([sum(Lj0/KLj0)])

GDP0= np.array([sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)])


# parameter definitions

sigmaXj=imp.sigmaXj

sigmaSj=imp.sigmaSj

etaXj=(imp.sigmaXj-1)/imp.sigmaXj

etaSj=(imp.sigmaSj-1)/imp.sigmaSj

alphaLj = imp.pLLj/imp.pKLjKLj

alphaKj = imp.pKKj/imp.pKLjKLj

bKLj=KLj0/(np.float_power(Lj0,alphaLj) * np.float_power(Kj0,alphaKj))

aYij= Yij0 / Yj0[None,:]

aKLj=KLj0/Yj0

alphaXj=np.float_power(division_by_zero(Yj0,Xj0),etaXj)*imp.pXjXj/imp.pYjYj

alphaDj=np.float_power(division_by_zero(Yj0,Dj0),etaXj)*imp.pDjDj/imp.pYjYj

betaMj=np.float_power(division_by_zero(Sj0,Mj0),etaSj)*imp.pMjMj/imp.pSjSj

betaDj=np.float_power(division_by_zero(Sj0,Dj0),etaSj)*imp.pDjDj/imp.pSjSj

alphaCj=imp.pCjCj/R0

alphaGj=imp.pCjGj/G0

alphaIj=imp.pCjIj/I0

alphalj=Lj0/(KLj0*l0)

wB = B0/GDP0

wG = G0/GDP0

wI = I0/GDP0

GDPreal=GDP0*(1+GDPgrowth)

pCjtp=pCj0

pXtp=pXj

Ctp=Cj0

Gtp=Gj0

Itp=Ij0

pXtp=pXj

Xtp=Xj0

Mtp = Mj0

betaRj= (imp.epsilonPCj+1)/(alphaCj-1)

epsilonRj=imp.epsilonRj

sD0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/GDP0

sKL0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/(pL0*L0+pK0*K0)



#CALIBRATE betaRj WITH REVENUE ELASTICITY OF CONSUMPTION (INSTEAD OF PRICE ELASTICITY) 

# def eqbetaRj(epsilonRj,betaRj,alphaCj):
#     zero= np.float_power(1 -epsilonRj / (1 + betaRj - sum(betaRj*alphaCj)),1)
#     return zero

# def eqepsilonRj(epsilonRjtarget,epsilonRj):
#     zero  = np.float_power(1 - epsilonRjtarget / epsilonRj,1)
#     return zero

# def  solvebetaRj(var, par):
    
#     d = {**var, **par}

#     return np.hstack([eqbetaRj(epsilonRj=d['epsilonRj'],betaRj=d['betaRj'],alphaCj=d['alphaCj']), 
#                       eqepsilonRj(epsilonRjtarget=d['epsilonRjtarget'], epsilonRj= d['epsilonRj'])
#                       ])

# variables={'betaRj':np.ones(N),
#             'epsilonRj': np.ones(N)}

# parameters={'epsilonRjtarget': imp.epsilonRj, 'alphaCj': alphaCj}

# bounds={    
#     'betaRj':(-1,np.inf),
#     'epsilonRj':(-np.inf,np.inf)
#     }


# def multiply_bounds_len(key,this_bounds,this_variables):
#     return [this_bounds[key] for i in range(len(this_variables[key].flatten()))]

# def bounds_dict(this_bounds,this_variables):
#     return dict((k, multiply_bounds_len(k,this_bounds,this_variables) ) for k in this_variables.keys())

# def flatten_bounds_dict(this_bounds,this_variables):
#     return np.vstack(list(bounds_dict(this_bounds,this_variables).values()))

# bounds_variables = [[row[i] for row in flatten_bounds_dict(bounds,variables)] for i in (0,1)]
    
# sol=dict_least_squares(solvebetaRj, variables, parameters, bounds_variables,verb=2)


# betaRj= sol.dvar['betaRj']

# epsilonRj=1 + sol.dvar['betaRj'] - sum(sol.dvar['betaRj']*alphaCj)


# dfR=pd.DataFrame({"betaRj":sol.dvar['betaRj'], "epsilonRj": sol.dvar['epsilonRj'],"epsilonRjComputed": epsilonRj , "epsilonRjtarget":imp.epsilonRj ,"difference":((epsilonRj-imp.epsilonRj)/imp.epsilonRj)*100 })

# dfR.to_csv("epsilonRj.csv")

# epsilonPCj=1-sol.dvar['betaRj']*(1-alphaCj)

# dfPC=pd.DataFrame({"betaRj":sol.dvar['betaRj'], "epsilonPCjComputed": epsilonPCj  , "epsilonPCjtarget":imp.epsilonPCj ,"difference":((epsilonPCj-imp.epsilonPCj)/abs(imp.epsilonPCj))*100 })

# dfPC.to_csv("epsilonPCj.csv")




######### calibrate alphaCDES (to rewrite!) ##################

def eqalphaCDES(alphaCj,alphaCDESj,R,pCj,betaRj):
    
    zero= 1- alphaCj/( alphaCDESj*np.float_power(R/pCj, betaRj)/sum(alphaCDESj*np.float_power(R/pCj,betaRj)) )
    return zero

def system(var, par):
    
    d = {**var, **par}

    return np.hstack([eqalphaCDES(alphaCj=d['alphaCj'],alphaCDESj=d['alphaCDESj'],R=d['R'],pCj=d['pCj'],betaRj=d['betaRj']),
                      1-sum(d['alphaCDESj'])
                      ])


variables={'alphaCDESj':alphaCj[alphaCj!=0]}

parameters={'alphaCj':alphaCj[alphaCj!=0],'R':R0,'pCj':pCj0[alphaCj!=0],'betaRj':betaRj[alphaCj!=0]}

bounds=np.array([([0]*len(alphaCj[alphaCj!=0])),([1]*len(alphaCj[alphaCj!=0]))])

solalpha=dict_least_squares(system, variables , parameters, bounds, check=False)

alphaCDESj=np.zeros(N)

alphaCDESj[alphaCj!=0]=solalpha.x

sum(alphaCDESj)

import numpy as np 
import import_GTAP_data as imp
from import_GTAP_data import N
from solvers import dict_least_squares
import sys
import copy
#import pandas as pd

tau=imp.labor_taxes / (imp.pLLj - imp.labor_taxes)
pL=1
w=pL/(1+tau)

def division_by_zero(num,den):
    if len(num)==len(den):
        n=len(num)
    else:
        print("denominator and numerator have different len")
        sys.exit()
        
    result=np.zeros(n)
    result[den!=0]=num[den!=0]/den[den!=0]
    return result

class calibrationVariables:
    def __init__(self, L_gr0, L0=None):
        
        #labor
        if L0 is None:   
            self.pL0 = 1
            self.Lj0= imp.pLLj / copy.deepcopy(self.pL0)
            self.L0=sum(copy.deepcopy(self.Lj0))
        else:
            self.L0=L0
            self.pL0=sum(imp.pLLj)/L0
            self.Lj0 = imp.pLLj / copy.deepcopy(self.pL0)
        
        #prezzi
        
        self.pYj0=np.array([float(1)]*N)
        self.pSj0=np.array([float(1)]*N)
        self.pKLj0=np.array([float(1)]*N)
        self.pXj0=np.array([float(1)]*N)
        self.pMj0=np.array([float(1)]*N)
        self.pDj0=np.array([float(1)]*N)
        self.pXj=np.array([float(1)]*N)
        
        #taxes
        
        self.tauYj0 = imp.production_taxes/( imp.pYjYj - imp.production_taxes)

        self.tauSj0 = imp.sales_taxes / (imp.pCiYij.sum(axis=1)+imp.pCjCj+imp.pCjGj+imp.pCjIj - imp.sales_taxes)
        self.tauL0 = imp.labor_taxes / (imp.pLLj - imp.labor_taxes)
        self.pCj0 = (1+copy.deepcopy(self.tauSj0))*copy.deepcopy(self.pSj0)
        self.w= copy.deepcopy(self.pL0)/(1+copy.deepcopy(self.tauL0))
        
        #quantità
        self.Ij0 = imp.pCjIj/ copy.deepcopy(self.pCj0)
        self.Cj0 = imp.pCjCj/ copy.deepcopy(self.pCj0)
        self.Gj0 = imp.pCjGj/ copy.deepcopy(self.pCj0)
        self.Yij0 = imp.pCiYij/ copy.deepcopy(self.pCj0[:,None])
        self.KLj0= imp.pKLjKLj / copy.deepcopy(self.pKLj0)
        self.Xj0= imp.pXjXj / copy.deepcopy(self.pXj0)
        self.Mj0= imp.pMjMj / copy.deepcopy(self.pMj0)
        self.Dj0= imp.pDjDj / copy.deepcopy(self.pDj0)
        self.Sj0= imp.pSjSj / copy.deepcopy(self.pSj0)
        self.Yj0= imp.pYjYj / copy.deepcopy(self.pYj0)
        
        #scalari
        
        self.T0= sum(imp.production_taxes + imp.sales_taxes + imp.labor_taxes)
        self.B0=sum(imp.pXjXj)-sum(imp.pMjMj)
        self.R0= sum(imp.pCjCj)
        self.Ri0= sum(imp.pCjIj)
        self.Rg0= sum(imp.pCjGj)
        self.l0=sum(copy.deepcopy(self.Lj0)/ copy.deepcopy(self.KLj0))
        self.uL0 = 0.105
        self.sigmaw= 0.
        self.uK0 = 0.105
        self.sigmapK= -0.1
        self.GDP0= sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)
        
        # parametri
        self.sigmaXj=imp.sigmaXj.astype(float)
        self.sigmaSj=imp.sigmaSj.astype(float)
        self.sigmaKLj=imp.sigmaKLj.astype(float)

        #self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        
        self.etaSj=(imp.sigmaSj-1)/imp.sigmaSj
        self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        self.etaKLj=(imp.sigmaKLj-1)/imp.sigmaKLj

        self.aYij= copy.deepcopy(self.Yij0) / copy.deepcopy(self.Yj0[None,:])
        self.aKLj= copy.deepcopy(self.KLj0)/ copy.deepcopy(self.Yj0)
        
        def compute_alphas_CES(Q1j,Q2j,p1j,p2j,etaj):
            alphaj = 1 / (
                1 + np.float_power( Q2j , (1 - etaj) ) * p2j / ( np.float_power( Q1j , 1-etaj ) * p1j )
                )
            return alphaj
        
        def compute_theta_CES(Zj,alpha1j,alpha2j,Q1j,Q2j,etaj):
            thetaj = Zj / ( 
                np.float_power( 
                    alpha1j * np.float_power(Q1j,etaj) + alpha2j * np.float_power( Q2j,etaj) ,
                    1/etaj )
                )
            
            return thetaj
        

        self.alphaXj= compute_alphas_CES(Q1j= copy.deepcopy(self.Xj0),Q2j= copy.deepcopy(self.Dj0),p1j= copy.deepcopy(self.pXj0),p2j= copy.deepcopy(self.pDj0),etaj= copy.deepcopy(self.etaXj))
        self.alphaDj= compute_alphas_CES(Q1j= copy.deepcopy(self.Dj0),Q2j= copy.deepcopy(self.Xj0),p1j= copy.deepcopy(self.pDj0),p2j= copy.deepcopy(self.pXj0),etaj= copy.deepcopy(self.etaXj))
        self.thetaj = compute_theta_CES(Zj= copy.deepcopy(self.Yj0), alpha1j= copy.deepcopy(self.alphaXj), alpha2j= copy.deepcopy(self.alphaDj), Q1j= copy.deepcopy(self.Xj0), Q2j= copy.deepcopy(self.Dj0),etaj= copy.deepcopy(self.etaXj))
        
        self.betaMj= compute_alphas_CES(Q1j= copy.deepcopy(self.Mj0),Q2j= copy.deepcopy(self.Dj0),p1j= copy.deepcopy(self.pMj0),p2j= copy.deepcopy(self.pDj0),etaj= copy.deepcopy(self.etaSj))
        self.betaDj= compute_alphas_CES(Q1j= copy.deepcopy(self.Dj0),Q2j= copy.deepcopy(self.Mj0),p1j= copy.deepcopy(self.pDj0),p2j= copy.deepcopy(self.pMj0),etaj= copy.deepcopy(self.etaSj))
        self.csij = compute_theta_CES(Zj= copy.deepcopy(self.Sj0),alpha1j= copy.deepcopy(self.betaMj),alpha2j= copy.deepcopy(self.betaDj),Q1j= copy.deepcopy(self.Mj0),Q2j= copy.deepcopy(self.Dj0),etaj= copy.deepcopy(self.etaSj))
        
        self.alphaCj = imp.pCjCj/ copy.deepcopy(self.R0)
        self.alphaGj = imp.pCjGj/ copy.deepcopy(self.Rg0)
        self.alphalj = copy.deepcopy(self.Lj0)/(copy.deepcopy(self.KLj0)*copy.deepcopy(self.l0))
        self.alphaw = copy.deepcopy(self.w)/(copy.deepcopy(self.uL0)**copy.deepcopy(self.sigmaw))
        self.wB = copy.deepcopy(self.B0)/ copy.deepcopy(self.GDP0)
        self.wG = copy.deepcopy(self.Rg0)/ copy.deepcopy(self.GDP0)
        self.wI = copy.deepcopy(self.Ri0)/ copy.deepcopy(self.GDP0)
        self.GDPreal= copy.deepcopy(self.GDP0)
        self.pCjtp= copy.deepcopy(self.pCj0)
        self.pXtp= copy.deepcopy(self.pXj)
        self.Ctp= copy.deepcopy(self.Cj0)
        self.Gtp= copy.deepcopy(self.Gj0)
        self.Itp= copy.deepcopy(self.Ij0)
        self.pXtp= copy.deepcopy(self.pXj)
        self.Xtp= copy.deepcopy(self.Xj0)
        self.Mtp = copy.deepcopy(self.Mj0)
        #self.betaRj= (imp.epsilonPCj+1)/(self.alphaCj-1)
        #self.epsilonRj=imp.epsilonRj
        self.sD0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/ copy.deepcopy(self.GDP0)
        
        #calibrate alphaIj, I and pI
        
        def eqpI(pI,pCj,alphaIj):
            zero= -1+ pI / sum(pCj*alphaIj)
            return zero
        
        def eqIj(Ij,alphaIj,I):
            zero= -1+Ij/(alphaIj*I)
            return zero
        
        def eqRi(Ri,pI,I):
            zero= - 1 + Ri / (pI*I)
            return zero
        
        def systemI(var, par):
            d = {**var, **par}
            return np.hstack([eqpI(pI=d['pI'],pCj=d['pCj'],alphaIj=d['alphaIj']),
                              eqIj(Ij=d['Ij'], alphaIj=d['alphaIj'],I=d['I']),
                              eqRi(Ri=d['Ri'],pI=d['pI'],I=d['I'])]
                              )
        
        this_len=len(imp.pCjIj[imp.pCjIj!=0])
        
        variables = { 'I': self.Ri0/sum(np.array([0.02]*N)*self.pCj0),
                   'alphaIj': np.array([10]*this_len),
                   'pI': np.array([sum(np.array([0.02]*N)*self.pCj0)])
                   }
        
        parameters={'pCj':self.pCj0[imp.pCjIj!=0],
                    'Ij':self.Ij0[imp.pCjIj!=0], 
                    'Ri':self.Ri0
                    }
        
        bounds= np.array([ ([0]*(this_len+2)),([np.inf]*(2+this_len)) ])
        
        solI = dict_least_squares(systemI, variables , parameters, bounds, N, check=False)
        
        self.I0=float(solI.dvar['I'])
        self.pI0=float(solI.dvar['pI'])
        self.alphaIj=np.zeros(N)
        self.alphaIj[imp.pCjIj!=0]=solI.dvar['alphaIj']
        self.delta=0.04
        self.g0=L_gr0
        self.pK0 = (sum(imp.pKKj)*(copy.deepcopy(self.g0)+copy.deepcopy(self.delta)))/ copy.deepcopy(self.I0)
        self.Kj0= imp.pKKj / copy.deepcopy(self.pK0)
        self.K0=sum(self.Kj0)
        self.alphapK = copy.deepcopy(self.pK0)/(copy.deepcopy(self.uK0)**copy.deepcopy(self.sigmapK))
        self.alphaIK = copy.deepcopy(self.Ri0)/ copy.deepcopy(self.K0)
        self.K0next = copy.deepcopy(self.K0) * (1-copy.deepcopy(self.delta)) + copy.deepcopy(self.I0)
        self.L0u=sum(self.Lj0)/(1-copy.deepcopy(self.uL0))
        self.K0u=sum(self.Kj0)/(1-copy.deepcopy(self.uK0))
        self.K0u_next= copy.deepcopy(self.K0u) * (1-copy.deepcopy(self.delta)) + copy.deepcopy(self.I0)
        self.GDPPI=1
        self.CPI=1
        self.alphaLj= compute_alphas_CES(Q1j= copy.deepcopy(self.Lj0),Q2j= copy.deepcopy(self.Kj0),p1j= copy.deepcopy(self.pL0),p2j= copy.deepcopy(self.pK0),etaj= copy.deepcopy(self.etaKLj))
        self.alphaKj= compute_alphas_CES(Q1j= copy.deepcopy(self.Kj0),Q2j= copy.deepcopy(self.Lj0),p1j= copy.deepcopy(self.pK0),p2j= copy.deepcopy(self.pL0),etaj= copy.deepcopy(self.etaKLj))
        self.gammaj = compute_theta_CES(Zj= copy.deepcopy(self.KLj0), alpha1j= copy.deepcopy(self.alphaKj), alpha2j= copy.deepcopy(self.alphaLj), Q1j= copy.deepcopy(self.Kj0), Q2j= copy.deepcopy(self.Lj0),etaj= copy.deepcopy(self.etaKLj))
        self.bKL=1
        self.bKLj = copy.deepcopy(self.KLj0)*copy.deepcopy(self.bKL)/np.float_power(copy.deepcopy(self.alphaLj)*np.float_power(copy.deepcopy(self.Lj0),copy.deepcopy(self.etaKLj)) + copy.deepcopy(self.alphaKj) * np.float_power(copy.deepcopy(self.Kj0),copy.deepcopy(self.etaKLj)), 1/ copy.deepcopy(self.etaKLj))


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

# def eqalphaCDES(alphaCj,alphaCDESj,R,pCj,betaRj):
    
#     zero= 1- alphaCj/( alphaCDESj*np.float_power(R/pCj, betaRj)/sum(alphaCDESj*np.float_power(R/pCj,betaRj)) )
#     return zero

# def system(var, par):
    
#     d = {**var, **par}

#     return np.hstack([eqalphaCDES(alphaCj=d['alphaCj'],alphaCDESj=d['alphaCDESj'],R=d['R'],pCj=d['pCj'],betaRj=d['betaRj']),
#                       1-sum(d['alphaCDESj'])
#                       ])


# variables={'alphaCDESj':alphaCj[alphaCj!=0]}

# parameters={'alphaCj':alphaCj[alphaCj!=0],'R':R0,'pCj':pCj0[alphaCj!=0],'betaRj':betaRj[alphaCj!=0]}

# bounds=np.array([ ( [0]*len(alphaCj[alphaCj!=0]) ) , ( [1]*len(alphaCj[alphaCj!=0]) ) ])

# solalpha=dict_least_squares(system, variables , parameters, bounds, check=False)




# alphaCDESj=np.zeros(N)

# alphaCDESj[alphaCj!=0]=solalpha.x

# sum(alphaCDESj)

#pSj0=np.array([100]*N)
#tauSj0 = imp.sales_taxes / (imp.pCiYij.sum(axis=1)+imp.pCjCj+imp.pCjGj+imp.pCjIj - imp.sales_taxes)
#pCj0 = (1+tauSj0)*pSj0

#def fun(x):
#    return 1 - (.1/ sum(x * pCj0[imp.pCjIj!=0]))

#this_len=len(imp.pCjIj[imp.pCjIj!=0])

#a=optimize.least_squares(
#    fun,
#    np.array([.1]*59),
#    bounds=np.array([ ([0]*(this_len)),([np.inf]*(this_len)) ])
#    )

#sum(a.x * pCj0[imp.pCjIj!=0])


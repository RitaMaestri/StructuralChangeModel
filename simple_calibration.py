import numpy as np 
import import_GTAP_data as imp
from import_GTAP_data import N
from solvers import dict_least_squares
import sys
from scipy import optimize
#import pandas as pd


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
            self.Lj0= imp.pLLj / self.pL0
            self.L0=sum(self.Lj0)
        else:
            self.L0=L0
            self.pL0=sum(imp.pLLj)/L0
            self.Lj0 = imp.pLLj / self.pL0
        
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
        self.pCj0 = (1+self.tauSj0)*self.pSj0
        self.w=self.pL0/(1+self.tauL0)
        
        #quantità
        
        self.Ij0 = imp.pCjIj/self.pCj0
        self.Cj0 = imp.pCjCj/self.pCj0
        self.Gj0 = imp.pCjGj/self.pCj0
        self.Yij0 = imp.pCiYij/self.pCj0[:,None]
        self.KLj0= imp.pKLjKLj / self.pKLj0
        self.Xj0= imp.pXjXj / self.pXj0
        self.Mj0= imp.pMjMj / self.pMj0
        self.Dj0= imp.pDjDj / self.pDj0
        self.Sj0= imp.pSjSj / self.pSj0
        self.Yj0= imp.pYjYj / self.pYj0
        
        #scalari
        
        self.T0= sum(imp.production_taxes + imp.sales_taxes + imp.labor_taxes)
        self.B0=sum(imp.pXjXj)-sum(imp.pMjMj)
        self.R0= sum(imp.pCjCj)
        self.Ri0= sum(imp.pCjIj)
        self.Rg0= sum(imp.pCjGj)
        self.l0=sum(self.Lj0/self.KLj0)
        self.uL0 = 0.105
        self.sigmaw= 0
        self.uK0 = 0.105
        self.sigmapK= -0.1
        self.GDP0= sum(imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)
        
        # parametri
        self.sigmaXj=imp.sigmaXj
        self.sigmaSj=imp.sigmaSj
        
        #self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        
        self.etaSj=(imp.sigmaSj-1)/imp.sigmaSj
        self.etaXj=(imp.sigmaXj-1)/imp.sigmaXj
        self.alphaLj = imp.pLLj/imp.pKLjKLj
        self.alphaKj = imp.pKKj/imp.pKLjKLj
        self.aYij= self.Yij0 / self.Yj0[None,:]
        self.aKLj=self.KLj0/self.Yj0
        
        self.alphaXj = np.float_power(division_by_zero(self.Yj0,self.Xj0),self.etaXj) * imp.pXjXj / imp.pYjYj
        self.alphaDj = np.float_power(division_by_zero(self.Yj0,self.Dj0),self.etaXj) * imp.pDjDj / imp.pYjYj
        
        self.betaMj = np.float_power(division_by_zero(self.Sj0,self.Mj0),self.etaSj)*imp.pMjMj / imp.pSjSj
        self.betaDj = np.float_power(division_by_zero(self.Sj0,self.Dj0),self.etaSj)*imp.pDjDj / imp.pSjSj
        self.alphaCj = imp.pCjCj/self.R0
        self.alphaGj = imp.pCjGj/self.Rg0
        self.alphalj = self.Lj0/(self.KLj0*self.l0)
        self.alphaw = self.w/(self.uL0**self.sigmaw)
        self.wB = self.B0/self.GDP0
        self.wG = self.Rg0/self.GDP0
        self.wI = self.Ri0/self.GDP0
        self.GDPreal=self.GDP0
        self.pCjtp=self.pCj0
        self.pXtp=self.pXj
        self.Ctp=self.Cj0
        self.Gtp=self.Gj0
        self.Itp=self.Ij0
        self.pXtp=self.pXj
        self.Xtp=self.Xj0
        self.Mtp = self.Mj0
        #self.betaRj= (imp.epsilonPCj+1)/(self.alphaCj-1)
        #self.epsilonRj=imp.epsilonRj
        self.sD0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/self.GDP0
        
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
#         self.I0=1395.532824265621 
#         self.pI0=2650.182180045942 
#         self.alphaIj=np.array([5.97780611e-05,2.31155359e-03,2.75605530e-03,9.40803591e-03
# ,1.75098464e-03,2.87839490e-04,2.97471645e-04,6.28461385e-03
# ,5.16597687e-03,7.67345637e-03,4.11191888e-03,1.01757817e-03
# ,6.06774907e-03,3.76377174e-04,1.51273351e-02,9.42553004e-05
# ,2.17355047e-04,1.67800489e-04,2.02527441e-04,6.04762054e-06
# ,4.62264142e-05,6.10731436e-04,1.70715559e-04,6.63490428e-03
# ,8.25267799e-03,3.58818494e-03,6.12709001e-02,1.91017131e-01
# ,3.43150646e-02,1.06485938e-02,1.14869095e-02,4.12068742e-02
# ,1.54471318e-02,1.40217225e-02,5.09509515e-01,9.41832685e-01
# ,7.50701076e-01,2.27230154e+00,1.02164439e+00,8.25383543e-01
# ,6.82781782e-01,4.27129323e-03,1.10683984e+01,1.13001813e+00
# ,3.86629510e-04,5.17996812e-02,4.84060959e-03,1.09745413e-03
# ,4.59309983e-03,2.61505618e+00,4.66758158e-03,1.67500269e-05
# ,1.90415287e-01,2.41307375e+00,1.28385948e-01,2.55922758e-02
# ,1.95683066e-02,2.07986623e-02,4.96593235e-07,0.00000000e+00
# ,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
# ,0.00000000e+00])
        self.delta=0.04
        self.g0=L_gr0
        self.pK0 = (sum(imp.pKKj)*(self.g0+self.delta))/self.I0
        self.Kj0= imp.pKKj / self.pK0
        self.K0=sum(self.Kj0)
        self.bKLj = self.KLj0/(np.float_power(self.Lj0,self.alphaLj) * np.float_power(self.Kj0,self.alphaKj))
        self.alphapK = self.pK0/(self.uK0**self.sigmapK)
        self.alphaIK = self.Ri0/self.K0
        self.K0next = self.K0 * (1-self.delta) + self.I0
        self.L0u=sum(self.Lj0)/(1-self.uL0)
        self.K0u=sum(self.Kj0)/(1-self.uK0)
        self.K0u_next= self.K0u * (1-self.delta) + self.I0

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


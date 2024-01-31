import numpy as np 
import import_GTAP_data as imp
from import_GTAP_data import N,sectors
from solvers import dict_least_squares
import sys
from copy import deepcopy as cp
import csv
#import pandas as pd

A = sectors.index("AGRICULTURE")
M = sectors.index("MANUFACTURE")
SE = sectors.index("SERVICES")
E = sectors.index("ENERGY")
ST = sectors.index("STEEL")
CH = sectors.index("CHEMICAL")
T = sectors.index("TRANSPORTATION")


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

def compute_intermediate_prices(idx_E, pCj, p_CEj):
        intermediate_prices=np.repeat( cp(pCj), len(pCj), axis=0 )
        intermediate_prices_matrix = intermediate_prices.reshape(len(pCj), len(pCj))
        

class calibrationVariables:
    def __init__(self, L_gr0, L0=None):
        
        #labor
        if L0 is None:   
            self.pL0 = 1
            self.Lj0= imp.pLLj / cp(self.pL0)
            self.L0=sum(cp(self.Lj0))
        else:
            self.L0=L0
            self.pL0=sum(imp.pLLj)/L0
            self.Lj0 = imp.pLLj / cp(self.pL0)
        

        
        #prezzi
        
        self.pYj0=np.array([float(5000)]*N)
        self.pSj0=np.array([float(5000)]*N)
        self.pKLj0=np.array([float(10000)]*N)
        self.pXj0=np.array([float(5000)]*N)
        self.pDj0=np.array([float(5000)]*N)
        self.pXj=np.array([float(5000)]*N)
        self.pMj0=cp(self.pXj0)
        #adjusting energy quantity and price
        self.Sj0= imp.pSjSj / cp(self.pSj0)
        self.Sj0[E]=91.9143818
        self.pSj0[E]=imp.pSjSj[E] / cp(self.Sj0[E])

        
        
        
        #taxes
        
        self.tauYj0 = imp.production_taxes/( imp.pYjYj - imp.production_taxes)

        self.tauSj0 = imp.sales_taxes / (imp.pCiYij.sum(axis=1)+imp.pCjCj+imp.pCjGj+imp.pCjIj - imp.sales_taxes)
        self.tauL0 = imp.labor_taxes / (imp.pLLj - imp.labor_taxes)
        self.pCj0 = (1+cp(self.tauSj0))*cp(self.pSj0)
        self.w= cp(self.pL0)/(1+cp(self.tauL0))
        
        #quantità
        self.Ij0 = imp.pCjIj/ cp(self.pCj0)
        self.Cj0 = imp.pCjCj/ cp(self.pCj0)
        self.Gj0 = imp.pCjGj/ cp(self.pCj0)

        self.Yij0 = imp.pCiYij/ cp(self.pCj0[:,None])
        self.KLj0= imp.pKLjKLj / cp(self.pKLj0)
        self.Xj0= imp.pXjXj / cp(self.pXj0)
        self.Mj0= imp.pMjMj / cp(self.pMj0)
        self.Dj0= imp.pDjDj / cp(self.pDj0)
        #self.Sj0= imp.pSjSj / cp(self.pSj0)
        self.Yj0= imp.pYjYj / cp(self.pYj0)

        #scalari

        self.T0= sum(imp.production_taxes + imp.sales_taxes + imp.labor_taxes)
        self.B0=sum(imp.pXjXj)-sum(imp.pMjMj)
        self.R0= sum(imp.pCjCj)
        self.Ri0= sum(imp.pCjIj)
        self.Rg0= sum(imp.pCjGj)
        self.l0=sum(cp(self.Lj0)/ cp(self.KLj0))
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

        
        self.aKLj= cp(self.KLj0)/ cp(self.Yj0)
        self.lambda_KL = 1
        
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
        

        self.alphaXj= compute_alphas_CES(Q1j= cp(self.Xj0),Q2j= cp(self.Dj0),p1j= cp(self.pXj0),p2j= cp(self.pDj0),etaj= cp(self.etaXj))
        self.alphaDj= compute_alphas_CES(Q1j= cp(self.Dj0),Q2j= cp(self.Xj0),p1j= cp(self.pDj0),p2j= cp(self.pXj0),etaj= cp(self.etaXj))
        self.thetaj = compute_theta_CES(Zj= cp(self.Yj0), alpha1j= cp(self.alphaXj), alpha2j= cp(self.alphaDj), Q1j= cp(self.Xj0), Q2j= cp(self.Dj0),etaj= cp(self.etaXj))
        
        self.betaMj= compute_alphas_CES(Q1j= cp(self.Mj0),Q2j= cp(self.Dj0),p1j= cp(self.pMj0),p2j= cp(self.pDj0),etaj= cp(self.etaSj))
        self.betaDj= compute_alphas_CES(Q1j= cp(self.Dj0),Q2j= cp(self.Mj0),p1j= cp(self.pDj0),p2j= cp(self.pMj0),etaj= cp(self.etaSj))
        self.csij = compute_theta_CES(Zj= cp(self.Sj0),alpha1j= cp(self.betaMj),alpha2j= cp(self.betaDj),Q1j= cp(self.Mj0),Q2j= cp(self.Dj0),etaj= cp(self.etaSj))
        
        self.alphaCj0 = imp.pCjCj/ cp(self.R0)
        self.lambda_E = 1
        self.lambda_nE = 1
        self.alphaGj = imp.pCjGj/ cp(self.Rg0)
        self.alphalj = cp(self.Lj0)/(cp(self.KLj0)*cp(self.l0))
        self.alphaw = cp(self.w)/(cp(self.uL0)**cp(self.sigmaw))
        self.wB = cp(self.B0)/ cp(self.GDP0)
        self.wG = cp(self.Rg0)/ cp(self.GDP0)
        self.wI = cp(self.Ri0)/ cp(self.GDP0)
        self.GDPreal= cp(self.GDP0)
        self.pXtp= cp(self.pXj)
        self.Gtp= cp(self.Gj0)
        self.Itp= cp(self.Ij0)
        self.pXtp= cp(self.pXj)
        self.Xtp= cp(self.Xj0)
        self.Mtp = cp(self.Mj0)
        #self.betaRj= (imp.epsilonPCj+1)/(self.alphaCj-1)
        #self.epsilonRj=imp.epsilonRj
        self.sD0=sum(imp.pCjIj+imp.pXjXj-imp.pMjMj)/ cp(self.GDP0)
        
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
        
        solI = dict_least_squares(systemI, variables , parameters, bounds, N, check=False,verb=0)
        
        self.I0=float(solI.dvar['I'])
        self.pI0=float(solI.dvar['pI'])
        self.alphaIj=np.zeros(N)
        self.alphaIj[imp.pCjIj!=0]=solI.dvar['alphaIj']
        self.delta=0.04
        self.g0=L_gr0
        self.pK0 = (sum(imp.pKKj)*(cp(self.g0)+cp(self.delta)))/ cp(self.I0)
        self.Kj0= imp.pKKj / cp(self.pK0)
        self.K0=sum(self.Kj0)
        self.alphapK = cp(self.pK0)/(cp(self.uK0)**cp(self.sigmapK))
        self.alphaIK = cp(self.Ri0)/ cp(self.K0)
        self.K0next = cp(self.K0) * (1-cp(self.delta)) + cp(self.I0)
        self.L0u=sum(self.Lj0)/(1-cp(self.uL0))
        self.K0u=sum(self.Kj0)/(1-cp(self.uK0))
        self.K0u_next= cp(self.K0u) * (1-cp(self.delta)) + cp(self.I0)
        self.GDPPI=1
        self.CPI=1
        self.alphaLj= compute_alphas_CES(Q1j= cp(self.Lj0),Q2j= cp(self.Kj0),p1j= cp(self.pL0),p2j= cp(self.pK0),etaj= cp(self.etaKLj))
        self.alphaKj= compute_alphas_CES(Q1j= cp(self.Kj0),Q2j= cp(self.Lj0),p1j= cp(self.pK0),p2j= cp(self.pL0),etaj= cp(self.etaKLj))
        self.gammaj = compute_theta_CES(Zj= cp(self.KLj0), alpha1j= cp(self.alphaKj), alpha2j= cp(self.alphaLj), Q1j= cp(self.Kj0), Q2j= cp(self.Lj0),etaj= cp(self.etaKLj))
        self.bKL=1
        self.bKLj = cp(self.KLj0)*cp(self.bKL)/np.float_power(cp(self.alphaLj)*np.float_power(cp(self.Lj0),cp(self.etaKLj)) + cp(self.alphaKj) * np.float_power(cp(self.Kj0),cp(self.etaKLj)), 1/ cp(self.etaKLj))
        
        
# _____ _   _ _____ ____   ______   __   ____ ___  _   _ ____  _     ___ _   _  ____ 
#| ____| \ | | ____|  _ \ / ___\ \ / /  / ___/ _ \| | | |  _ \| |   |_ _| \ | |/ ___|
#|  _| |  \| |  _| | |_) | |  _ \ V /  | |  | | | | | | | |_) | |    | ||  \| | |  _ 
#| |___| |\  | |___|  _ <| |_| | | |   | |__| |_| | |_| |  __/| |___ | || |\  | |_| |
#|_____|_| \_|_____|_| \_\\____| |_|    \____\___/ \___/|_|   |_____|___|_| \_|\____|
#                                                                                    
        
        #i don't have to determine pC_E because it is endogenously determined. 
        sE_P = 0.196684710770692#from excel
        sE_T = 0.187353170012835
        sE_B = 0.194478926474159
        sY_E_PE = 1-(sE_P+sE_T+sE_B)
        
        S_E=cp(self.Sj0[E])
        self.E_P = sE_P*S_E
        self.E_B= sE_B*S_E
        self.E_T=sE_T*S_E
        self.Y_EE = sY_E_PE * S_E
        
        #PRIMARY ENERGY
        self.YE_Ej = np.array([float(0)]*N)
        self.YE_Ej[E] = self.Y_EE
        
        #ENERGY FOR PROCESSES
        self.YE_Pj = np.array([float(0)]*N)
        self.YE_Pj[A]=0.0617630630154619*cp(self.E_P)
        self.YE_Pj[CH]=0.346392415871497*cp(self.E_P)
        self.YE_Pj[ST]=0.154797014226055*cp(self.E_P)
        self.YE_Pj[M]=0.437047506886987*cp(self.E_P)

        
        #ENERGY FOR TRANSPORT
        self.YE_Tj = np.array([float(0)]*N)        
        self.s_LDV = 0.459509001531155
        self.s_trucks = 0.227480975484387
        self.s_other = 0.313010022984458
        
        self.YE_Tj[T]=(self.s_LDV*0.009+self.s_trucks*0.65+self.s_other)*cp(self.E_T)
        self.C_ET = 0.9 * (self.s_LDV * cp(self.E_T))
        nonT = [A,M,SE,ST,CH,E]
        for i in nonT:    
            self.YE_Tj[i]=cp(self.KLj0[i]) / cp(self.KLj0[nonT].sum()) * (cp(self.E_T)-cp(self.YE_Tj[T])-cp(self.C_ET)) 
        
        #ENERGY FOR BUILDINGS
        self.YE_Bj = np.array([float(0)]*(N))
        sC_EB = 0.65
        self.C_EB = sC_EB*cp(self.E_B)
        self.YE_Bj[SE]= (1-sC_EB)*cp(self.E_B)
        
        #energy volumes
        self.Cj0[E] = cp(self.C_EB)+cp(self.C_ET)
        for j in [A,M,CH,ST]:
            self.Yij0[E,j] =  cp(self.YE_Pj[j]) +  cp(self.YE_Tj[j])
        self.Yij0[E,SE] =  cp(self.YE_Bj[SE]) +  cp(self.YE_Tj[SE])
        self.Yij0[E,T] =  cp(self.YE_Tj[T])
        self.Yij0[E,E] =  cp(self.YE_Ej[E]) + cp(self.YE_Tj[E])
        
        ############ ENERGY PRICES   #############
        
        #consumption
        self.pY_Ej = np.array([float(0)]*(N))
        for i in [A,M,SE,ST,CH,E,T]:
            self.pY_Ej[i] = imp.pCiYij[E,i]/cp(self.Yij0[E,i])
        # self.pY_Ej[-1] = imp.pCjCj[E]/cp(self.Cj0[E])
        
        self.pCj0[E] = imp.pCjCj[E]/cp(self.Cj0[E])
        
        #transport
        self.pE_TT = self.pY_Ej[T]
        self.pE_TnT = (( cp(self.C_EB)*imp.pCiYij[E,SE] - imp.pCjCj[E] * cp(self.YE_Bj[SE]) ) /
                 ( cp(self.C_EB) * cp(self.YE_Tj[SE]) - cp(self.C_ET) * cp(self.YE_Bj[SE] ) ) )
        
        #buildings
        self.pE_B = 1 / cp(self.C_EB) * ( imp.pCjCj[E] - cp(self.C_ET) * cp(self.pE_TnT) )
        
        #processes
        self.pE_Pj = np.array([float(0)]*(N))
        for i in [A,M,ST,CH]:
            self.pE_Pj[i] = ( imp.pCiYij[E,i] - cp(self.pE_TnT) * cp(self.YE_Tj[i]) ) / cp(self.YE_Pj[i])
        
        #primary energy
        self.pE_Ej = np.array([float(0)]*(N))
        self.pE_Ej[E] = (imp.pCiYij[E,E] - cp(self.pE_TnT) * cp(self.YE_Tj[E]) ) / cp(self.Y_EE)
        

        #adjusted variables
        self.aYij= cp(self.Yij0) / cp(self.Yj0[None,:])
        
        self.pCjtp= cp(self.pCj0)
        self.Ctp= cp(self.Cj0)
        
        self.lambda_KLM = np.array([float(1)]*(N))
        
        self.aKLj0=cp(self.aKLj)
        self.aYij0=cp(self.aYij)
        
        
        



a=calibrationVariables(0.003985893420850095)


export_calib_dict={
        "YE_T_A" : a.YE_Tj[A],
        "YE_T_M" : a.YE_Tj[M],
        "YE_T_SE" : a.YE_Tj[SE],
        "YE_T_E" : a.YE_Tj[E],
        "YE_T_ST" : a.YE_Tj[ST],
        "YE_T_CH" : a.YE_Tj[CH],
        "YE_T_T" : a.YE_Tj[T],
        "C_ET": a.C_ET,
        "E_T":a.E_T,
        "s_LDV": a.s_LDV,
        "s_trucks" : a.s_trucks,
        "s_other" : a.s_other,
    }

with open('transport_calibration.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, export_calib_dict.keys())
    w.writeheader()
    w.writerow(export_calib_dict)




















#specific margins
#self.tau_Ej =   cp(self.pY_Ej) / cp(self.pCj0[E]) - 1

#self.smj = np.array([float(0)]*(N+1))

#for i in [A,M,SE,ST,CH,E,T]:
#    self.smj[i] = cp(self.tau_Ej[i])*cp(self.Yij0[E,i])
#self.smj[-1]=cp(self.tau_Ej[-1])*cp(self.Cj0[E])
        


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


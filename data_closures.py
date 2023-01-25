import numpy as np 
import simple_calibration as cal
import sys

N=cal.N




#######  Class Variable  #############

class Variable:
    def __init__(self, status, value):
        self.status=status
        self.value=value
   
    def __call__(self):
            list(self.status, self.value)



######## Class variableSystem ###########


class variablesSystem:
    def assignClosure(self, commonDict):
        
        sKLneoclassic = (cal.I0+cal.B0)/(cal.w * sum(cal.Lj0) + cal.pK0 * sum(cal.Kj0))
        
        sLkaldorian = (cal.I0+cal.B0)/(cal.w*sum(cal.Lj0))
        
        L0u=sum(cal.Lj0)/(1-cal.uL0)
        
        K0u=sum(cal.Kj0)/(1-cal.uK0)

        if self.closure == "johansen":
            return {**commonDict, 
                         **{'K':Variable("exo", cal.K0),
                            'sD':Variable("endo", cal.sD0),
                            'wI':Variable("exo",cal.wI),
                            'L':Variable("exo", cal.L0)}
                         }
        elif self.closure == "neoclassic":
            return {**commonDict, 
                         **{'K':Variable("exo", cal.K0),
                             'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", np.array([0])),
                            'L':Variable("exo", cal.L0)}
                         }
        
        
        elif self.closure == "kaldorian":
            return {**commonDict, 
                         **{'K':Variable("exo", cal.K0),
                             'l':Variable("endo", cal.l0),
                            'alphalj':Variable("exo", cal.alphalj),
                            'sK':Variable("exo", np.array([0])),
                            'sL':Variable("exo", sLkaldorian),
                            'sG':Variable("exo", np.array([0])),
                            'wI':Variable("exo",cal.wI),
                            'L':Variable("exo", cal.L0)
                            }
                         }
        
        
        elif self.closure == "keynes-marshall":
            return {**commonDict, 
                         **{'K':Variable("exo", cal.K0),
                            'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", np.array([0])),
                            'wI':Variable("exo",cal.wI),
                            }
                         }
        
        
        elif self.closure == "keynes-kaldor":
            return {**commonDict, 
                         **{'K':Variable("exo", cal.K0),
                             'l':Variable("endo", cal.l0),
                            'alphalj':Variable("exo", cal.alphalj),
                            'sK':Variable("exo", np.array([0])),
                            'sL':Variable("exo", sLkaldorian),
                            'sG':Variable("exo", np.array([0])),
                            'wI':Variable("exo",cal.wI),
                            
                            'w_real':Variable("endo",np.array([cal.w])),
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),
                            'uL':Variable("endo",cal.u0),
                            'L':Variable("exo", L0u)
                            
                            }
                         }
        
        
        elif self.closure == "keynes":
            return {**commonDict, 
                         **{ 'K':Variable("exo", cal.K0),
                            
                             'l':Variable("endo", cal.l0),
                            'alphalj':Variable("exo", cal.alphalj),
                            
                            'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", np.array([0])),
                            
                            'wI':Variable("exo",cal.wI),
                            
                            'w_real':Variable("endo",np.array([cal.w])),
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),
                            'uL':Variable("endo",cal.uL0),
                            'L':Variable("exo", L0u)
                            }
                         }
        
        elif self.closure == "neokeynesian1":
            return {**commonDict, 
                         **{'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", np.array([0])),                            
                            
                            'w_real':Variable("endo",np.array([cal.w])),
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),
                            'uL':Variable("endo",cal.uL0),
                            'L':Variable("exo", L0u),
                            
                            'pK_real':Variable("endo",np.array([cal.pK0])),
                            'alphapK':Variable("exo", cal.alphapK),
                            'sigmapK':Variable("exo", cal.sigmapK),
                            'uK':Variable("endo",cal.uK0),
                            'K':Variable("exo", K0u),
                            
                            'alphaIK':Variable("exo", cal.alphaIK)
                            }
                    }


        else: 
            print("this closure doesn't exist")
            sys.exit()


            
    def toEndoexoDict(self, status):
        return { k : v.value for k,v in self.variables_dict.items() if v.status == status }
        
    
    def __init__(self, closure):
        commonDict={
            'pL': Variable("endo", np.array([cal.pL0])),
            'pK':Variable("endo", np.array([cal.pK0])),
            'B':Variable("endo", np.array(cal.B0)),
            'R':Variable("endo", np.array(cal.R0)),
            'I':Variable("endo", np.array(cal.I0)),
            'G':Variable("endo", np.array(cal.G0)),
            'bKL':Variable("endo", np.array([1])),
            'CPI':Variable("endo", np.array([1])),
            'Rreal':Variable("endo", np.array(cal.R0)),
            'GDPPI':Variable("endo", np.array([1])),
            'GDP':Variable("endo", cal.GDPreal),
            'T':Variable("endo", cal.T0),
            'w':Variable("endo", np.array([cal.w])),
            'Kj':Variable("endo", cal.Kj0),
            'Lj':Variable("endo", cal.Lj0),
            'KLj':Variable("endo", cal.KLj0),
            'pKLj':Variable("endo", cal.pKLj0),
            'Yj':Variable("endo", cal.Yj0),
            'pYj':Variable("endo", cal.pYj0),
            'Cj': Variable("endo", cal.Cj0),
            'pCj': Variable("endo", cal.pCj0),
            'Mj': Variable("endo", cal.Mj0),
            'pMj': Variable("endo", cal.pMj0),
            'Xj': Variable("endo", cal.Xj0),
            'Dj': Variable("endo", cal.Dj0),
            'pDj': Variable("endo", cal.pDj0),
            'Sj': Variable("endo", cal.Sj0),
            'pSj': Variable("endo", cal.pSj0),
            'Gj': Variable("endo",cal.Gj0),
            'Ij': Variable("endo", cal.Ij0),
            'Yij': Variable("endo", cal.Yij0),
            
            
            'wG':Variable("exo", cal.wG),
            'wB':Variable("exo", cal.wB),
            'GDPreal':Variable("exo",cal.GDPreal ),
            'tauL':Variable("exo", np.array([cal.tauL0])),
            'tauSj':Variable("exo", cal.tauSj0),
            'tauYj':Variable("exo", cal.tauYj0),
            'pXj':Variable("exo", cal.pXj),
            'bKLj':Variable("exo", cal.bKLj),
            'pCtp':Variable("exo", cal.pCjtp),
            'Ctp':Variable("exo", cal.Ctp),
            'Gtp':Variable("exo", cal.Gtp),
            'Itp':Variable("exo", cal.Itp),
            'pXtp':Variable("exo", cal.pXtp),
            'Xtp':Variable("exo", cal.Xtp),
            'Mtp':Variable("exo", cal.Mtp),
            'alphaKj':Variable("exo", cal.alphaKj),
            'alphaLj':Variable("exo", cal.alphaLj),
            'aKLj':Variable("exo", cal.aKLj),
            'alphaCj':Variable("exo", cal.alphaCj),
            'alphaGj':Variable("exo", cal.alphaGj),
            'alphaIj':Variable("exo", cal.alphaIj),
            'alphaXj':Variable("exo", cal.alphaXj),
            'alphaDj':Variable("exo", cal.alphaDj),
            'betaDj':Variable("exo", cal.betaDj),
            'betaMj':Variable("exo", cal.betaMj),
            'sigmaXj':Variable("exo", cal.sigmaXj),
            'sigmaSj':Variable("exo", cal.sigmaSj),
            'aYij':Variable("exo", cal.aYij),
            }
        
        self.closure=closure
        
        self.variables_dict = self.assignClosure(commonDict)
        
        self.endogeouns_dict = self.toEndoexoDict("endo")
        
        self.exogenous_dict = self.toEndoexoDict("exo")
    
        



###########   BOUNDS    ################


bounds={
    'Rreal':(0,np.inf),
    'CPI':(-np.inf,np.inf),
    'pL':(0,np.inf),
    'pK':(0,np.inf),
    'w':(0,np.inf),
    'K':(0,np.inf),
    'L':(0,np.inf),
    'B':(-np.inf,np.inf),
    'wB':(-1,1),
    'GDPPI':(-np.inf,np.inf),
    'GDPreal':(-np.inf,np.inf), 
    'GDP':(-np.inf,np.inf),
    'R':(0,np.inf),
    'I':(0,np.inf),
    'G':(0,np.inf),
    'l':(0,np.inf),
    'bKL':(-np.inf,np.inf),
    'tauL':(0,1),
    'tauSj':(-1,1),
    'tauYj':(-1,1),
    'T':(-np.inf,np.inf),
    'w_real':(0,np.inf),
    'uL':(0,1),
    'sigmaw':(-np.inf,np.inf),
    'pK_real':(0,np.inf),
    'alphapK':(0,np.inf),
    'sigmapK':(-np.inf,np.inf),
    'uK':(0,1),
    
    'bKLj':(0,np.inf),    
    'Cj':(0,np.inf),
    'pCj':(0,np.inf),
    'Sj':(0,np.inf),
    'pSj':(0,np.inf),
    'Kj':(0,np.inf),
    'Lj':(0,np.inf),
    'KLj':(0,np.inf),
    'pKLj':(0,np.inf),
    'Dj':(0,np.inf),
    'pDj':(0,np.inf),
    'Xj':(0,np.inf),
    'pXj':(0,np.inf),
    'Yj':(0,np.inf),
    'pYj':(0,np.inf),
    'Mj':(0,np.inf),
    'pMj':(0,np.inf),  
    'Gj':(0,np.inf),
    'Ij':(0,np.inf),
    'pCtp':(0,np.inf),
    'Ctp':(0,np.inf),
    'Gtp':(0,np.inf),
    'Itp':(0,np.inf),
    'pXtp':(0,np.inf),
    'Xtp':(0,np.inf),
    'Mtp':(0,np.inf),
    'alphaKj':(0,1),
    'alphaLj':(0,1),
    'aKLj':(0,np.inf),
    'alphaCj':(0,1),
    'alphaXj':(0,np.inf),
    'alphaDj':(0,np.inf),
    'alphaGj':(0,np.inf),
    'alphaIj':(0,np.inf),    
    'alphalj':(0,1),  
    'alphaw':(0,np.inf),  
    'alphaIK':(0,np.inf),  
    'betaDj':(0,np.inf),
    'betaMj':(0,np.inf),
    'sigmaXj':(-np.inf,np.inf),
    'sigmaSj':(-np.inf,np.inf),
    'wG':(0,1),
    'wI':(0,1),
    'sD':(0,1),
    'sK':(0,1),
    'sL':(0,1),
    'sG':(0,1),
    'aYij':(0,np.inf),
    'Yij':(0,np.inf),
        }


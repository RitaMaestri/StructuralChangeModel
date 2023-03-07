import numpy as np 
from simple_calibration import calibrationVariables
from import_csv import N
import pandas as pd
import sys


#######  Class Variable  #############

class Variable:
    def __init__(self, status, value):
        self.status=status
        self.value=value
   
    def __call__(self):
            list(self.status, self.value)



######## Class variableSystem ###########


class calibrationDict:
    
    
    def assignClosure(self,cal):
        
        sKLneoclassic = (cal.Ri0+cal.B0)/(cal.w * sum(cal.Lj0) + cal.pK0 * sum(cal.Kj0))
        
        sLkaldorian = (cal.Ri0+cal.B0)/(cal.w*sum(cal.Lj0))
        
        if self.closure == "johansen":
            return {**self.commonDict, 
                         **{'sD':Variable("endo", cal.sD0),
                            
                            'K':Variable("exo", cal.K0),
                            'wI':Variable("exo",cal.wI),
                            'wB':Variable("exo", cal.wB),
                            'L':Variable("exo", cal.L0)
                            }
                         }
        
        elif self.closure == "neoclassic":
            return {**self.commonDict, 
                         **{'K':Variable("exo", cal.K0),
                             'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", 0),
                            'wB':Variable("exo", cal.wB),
                            'L':Variable("exo", cal.L0)}
                         }
        
        elif self.closure == "kaldorian":
            return {**self.commonDict, 
                         **{'l':Variable("endo", cal.l0),
                            
                            'K':Variable("exo", cal.K0),
                            'alphalj':Variable("exo", cal.alphalj),
                            'sK':Variable("exo", 0),
                            'sL':Variable("exo", sLkaldorian),
                            'sG':Variable("exo", 0),
                            'wI':Variable("exo",cal.wI),
                            'wB':Variable("exo", cal.wB),
                            'L':Variable("exo", cal.L0)
                            }
                         }
        
        elif self.closure == "keynes-marshall":
            return {**self.commonDict, 
                         **{'K':Variable("exo", cal.K0),
                            'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", 0),
                            'wB':Variable("exo", cal.wB),
                            'wI':Variable("exo",cal.wI),
                                  
                            }
                         }
        
        elif self.closure == "keynes-kaldor":
            return {**self.commonDict, 
                         **{ 'l':Variable("endo", cal.l0),
                            'w_real':Variable("endo",cal.w),
                            'uL':Variable("endo",cal.uL0),
                            
                            'K':Variable("exo", cal.K0),
                            'alphalj':Variable("exo", cal.alphalj),
                            'sK':Variable("exo", 0),
                            'sL':Variable("exo", sLkaldorian),
                            'sG':Variable("exo", 0),
                            'wI':Variable("exo",cal.wI),
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),
                            'wB':Variable("exo", cal.wB),
                            'L':Variable("exo", cal.L0u)
                            
                            }
                         }
        
        elif self.closure == "keynes":
            return {**self.commonDict, 
                         **{ 'l':Variable("endo", cal.l0),
                            'w_real':Variable("endo",cal.w),
                            'uL':Variable("endo",cal.uL0),
                            
                            'K':Variable("exo", cal.K0),
                            'alphalj':Variable("exo", cal.alphalj),
                            'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", 0),
                            'wI':Variable("exo",cal.wI),
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),
                            'wB':Variable("exo", cal.wB),
                            'L':Variable("exo", cal.L0u)
                            }
                         }
        
        elif self.closure == "neokeynesian1":
            return {**self.commonDict, 
                         **{'uL':Variable("endo",cal.uL0),
                            'pK_real':Variable("endo",cal.pK0),
                            'uK':Variable("endo",cal.uK0),
                            'w_real':Variable("endo",cal.w),
                             
                             
                            'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", 0),                            
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),
                            'L':Variable("exo", cal.L0u),
                            'alphapK':Variable("exo", cal.alphapK),
                            'sigmapK':Variable("exo", cal.sigmapK),
                            'K':Variable("exo", cal.K0u),
                            'wB':Variable("exo", cal.wB),
                            
                            }
                    }

        elif self.closure == "neokeynesian2":
            return {**self.commonDict, 
                         **{'w_real':Variable("endo",cal.w),
                            'uL':Variable("endo",cal.uL0),
                            'pK_real':Variable("endo",cal.pK0),
                            'uK':Variable("endo",cal.uK0),
                            
                            'sK':Variable("exo", sKLneoclassic),
                            'sL':Variable("exo", sKLneoclassic),
                            'sG':Variable("exo", 0),                            
                            'alphaw':Variable("exo", cal.alphaw),
                            'sigmaw':Variable("exo", cal.sigmaw),                            
                            'L':Variable("exo", cal.L0u),
                            'alphapK':Variable("exo", cal.alphapK),
                            'sigmapK':Variable("exo", cal.sigmapK),
                            'K':Variable("exo", cal.K0u),
                            'alphaIK':Variable("exo", cal.alphaIK),
                            
                            }
                    }
        else: 
            print("this closure doesn't exist")
            sys.exit()

    def toEndoExoDict(self, status):
        return { k : v.value for k,v in self.variables_dict.items() if v.status == status }


    def __init__(self, closure,initial_L_gr, endoKnext):
        cal = calibrationVariables(initial_L_gr)
        
        self.commonDict = {
            'pL': Variable("endo", cal.pL0),
            'pK':Variable("endo", cal.pK0),
            'pI':Variable("endo", cal.pI0),
            'B':Variable("endo", cal.B0),
            'R':Variable("endo", cal.R0),
            'Ri':Variable("endo", cal.Ri0),
            'Rg':Variable("endo", cal.Rg0),
            'bKL':Variable("endo", 1),
            'CPI':Variable("endo", 1),
            'Rreal':Variable("endo", cal.R0),
            'GDPPI':Variable("endo", 1),
            'GDP':Variable("endo", cal.GDPreal),
            'T':Variable("endo", cal.T0),
            'w':Variable("endo", cal.w),
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
            'I':Variable("endo", cal.I0),
            
            'wG':Variable("exo", cal.wG),
            'GDPreal':Variable("exo",cal.GDPreal ),
            'tauL':Variable("exo", cal.tauL0),
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
            'delta':Variable("exo", cal.delta)
            }
        
        Knext = Variable("endo", cal.K0u_next) if closure in ["neokeynesian1","neokeynesian2"] else Variable("endo", cal.K0next)
        
        self.commonDict = {**self.commonDict, **{'Knext': Knext}} if endoKnext else self.commonDict
        
        self.closure=closure
        
        self.variables_dict = self.assignClosure(cal)
        
        self.endogeouns_dict = self.toEndoExoDict("endo")
        
        self.exogenous_dict = self.toEndoExoDict("exo")

        



###########   BOUNDS    ################


bounds={
    'Rreal':(0,np.inf),
    'CPI':(-np.inf,np.inf),
    'pL':(0,np.inf),
    'pK':(0,np.inf),
    'pI':(0,np.inf),
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
    'Ri':(0,np.inf),
    'Rg':(0,np.inf),
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
    'Knext':(0,np.inf),
    
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
    'delta':(0,1)
        }


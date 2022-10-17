import numpy as np
import simple_calibration as cal
import import_csv as imp

#CAUTION: if you use the solver dict_minimize, 
#the bounds for the variables are (0,1) for variables whose initial guess is <1, (0,inf) for the rest
#if you use least_square it's (0,inf) in any case
N=cal.N

variables = {
    'pL':np.array([1]),
    'pK':np.array([1]),
    'B':np.array(cal.B0),
    'R':np.array(cal.R0),
    'bKL':np.array([1]),
    'GDP':cal.GDPreal,
    'GDPPI':np.array([1]),


    'Kj':imp.pKKj,
    'Lj':imp.pLLj,
    
    'KLj':cal.KLj0,
    'pKLj':cal.pKLj0,
    
    'Yj':cal.Yj0,
    'pYj':cal.pYj0, 
    
    'Cjn0':cal.Cj0[cal.Cj0!=0],
    'pCj':cal.pCj0, 
    
    'Mjn0':cal.Mj0[cal.Mj0!=0],
    'pMj':cal.pMj0,
    
    'Xjn0':cal.Xj0[cal.Xj0!=0],
    
    'Dj':cal.Dj0, 
    'pDj':cal.pDj0,

    'Sj':cal.Sj0,
    'pSj':cal.pCj0,

    'Gjn0':cal.Gj0[cal.Gj0!=0],
    'Ijn0':cal.Ij0[cal.Ij0!=0],
    
    'Yijn0':cal.Yij0[cal.Yij0!=0],
    }


parameters= {
    'L':np.array([sum(cal.L)]),
    'K':np.array([sum(cal.K)]),
    'wB':cal.wB,
    'GDPreal':cal.GDPreal,


    'pXj':np.array([100]*N),
    'pCtp':cal.pCjtp,
    'Ctp':cal.Ctp,
    'Gtp':cal.Gtp,
    'Itp':cal.Itp,
    'alphaKj':cal.alphaKj,
    'alphaLj':cal.alphaLj,
    'aKLj':cal.aKLj,
    'alphaCj':cal.alphaCj,
    'alphaXj':cal.alphaXj,
    'alphaDj':cal.alphaDj,
    'betaDj':cal.betaDj,
    'betaMj':cal.betaMj,
    'sigmaXj':cal.sigmaXj,
    'sigmaSj':cal.sigmaSj,
    'wGj':cal.wGj,
    'wIj':cal.wIj,
    
    'aYij':cal.aYij
    }




bounds={    
    'pL':(0,np.inf),
    'pK':(0,np.inf),
    'L':(0,np.inf),
    'K':(0,np.inf),
    'B':(-np.inf,np.inf),
    'wB':(-1,1),
    'GDPPI':(-np.inf,np.inf),
    'GDPreal':(-np.inf,np.inf), 
    'GDP':(-np.inf,np.inf),
    'bKL':(0,np.inf),
    'R':(0,np.inf),

    'Cjn0':(0,np.inf),
    'pCj':(0,np.inf),
    'Sj':(0,np.inf),
    'pSj':(0,np.inf),
    'Kj':(0,np.inf),
    'Lj':(0,np.inf),
    'KLj':(0,np.inf),
    'pKLj':(0,np.inf),
    'Dj':(0,np.inf),
    'pDj':(0,np.inf),
    'Xjn0':(0,np.inf),
    'pXj':(0,np.inf),
    'Yj':(0,np.inf),
    'pYj':(0,np.inf),
    'Mjn0':(0,np.inf),
    'pMj':(0,np.inf),  
    'Gjn0':(0,np.inf),
    'Ijn0':(0,np.inf),
    'pCtp':(0,np.inf),
    'Ctp':(0,np.inf),
    'Gtp':(0,np.inf),
    'Itp':(0,np.inf),
    'alphaKj':(0,1),
    'alphaLj':(0,1),
    'aKLj':(0,1),
    'alphaCj':(0,1),
    'alphaXj':(0,np.inf),
    'alphaDj':(0,np.inf),
    'betaDj':(0,np.inf),
    'betaMj':(0,np.inf),
    'sigmaXj':(-np.inf,np.inf),
    'sigmaSj':(-np.inf,np.inf),
    'wGj':(-1,1),
    'wIj':(-1,1),
    
    'aYij':(0,np.inf),
    'Yijn0':(0,np.inf),
        }



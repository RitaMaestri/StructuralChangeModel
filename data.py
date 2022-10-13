import numpy as np

#CAUTION: if you use the solver dict_minimize, 
#the bounds for the variables are (0,1) for variables whose initial guess is <1, (0,inf) for the rest
#if you use least_square it's (0,inf) in any case

variables = {
    'pL':np.array([1]),
    'pK':np.array([4]),
    'B':np.array([100]),
    'R':np.array([.2]),
    'bKL':np.array([.2]),
    #'GDPPI':np.array([1]),

    
    'Kj':np.array([100,200,150]),
    'Lj':np.array([120,50,140]),
    
    'KLj':np.array([200,300,320]),
    'pKLj':np.array([4,7,5]),
    
    'Yj':np.array([300,320,450]),
    'pYj':np.array([4,5,7]), 
    
    'Cj':np.array([570,450,620]),
    'pCj':np.array([5,6,8]), 
    
    'Mj':np.array([408,308,400]),
    'pMj':np.array([4,2,9]),
    
    'Xj':np.array([300,400,500]),
    
    'Dj':np.array([438,378,730]), 
    'pDj':np.array([4,2,8]),

    'Sj':np.array([438,378,650]),
    'pSj':np.array([4,3,4]),

    'Gj':np.array([200,200,200]),
    'Ij':np.array([200,200,200]),
    
    'Yij':np.array([[148,157,400],[202,265,300],[202,265,300]]),
    }


parameters= {
    'L':np.array([111]),
    'K':np.array([111]),
    'wB':np.array([.1]),
    #'GDPreal':np.array([2000]), 
    'GDP':np.array([4000]),

    'pXj':np.array([10,10,15]),
    #'pCtp':np.array([50, 34.6692715,24 ]),
    #'Ctp':np.array([ 7.95933127, 15.69112867,13]),
    #'Gtp':np.array([2.92622473, 1.44219933,1.5]),
    #'Itp':np.array([3.51146968, 1.73063919,2.5]),
    'alphaKj':np.array([.1,.6,.3]),
    'alphaLj':np.array([.2,.7,.1]),
    'aKLj':np.array([0.2,0.3,.4]),
    'alphaCj':np.array([0.2,0.3,.5]),
    'alphaXj':np.array([0.2,0.5,.3]),
    'alphaDj':np.array([0.2,0.2,.6]),
    'betaDj':np.array([0.1,0.2,.7]),
    'betaMj':np.array([0.3,0.3,.4]),
    'sigmaXj':np.array([0.5,0.1,.4]),
    'sigmaSj':np.array([-.8,-0.8,-.8]),
    'wGj':np.array([0.05,0.05,.05]),
    'wIj':np.array([0.06,0.06,.06]),
    
    'aYij':np.array([[0.2,0.1,.2],[0.3,0.2,.2],[.3,.4,.2]])
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
    'alphaKj':(0,1),
    'alphaLj':(0,1),
    'aKLj':(0,1),
    'alphaCj':(0,1),
    'alphaXj':(0,1),
    'alphaDj':(0,1),
    'betaDj':(0,1),
    'betaMj':(0,1),
    'sigmaXj':(-np.inf,np.inf),
    'sigmaSj':(-np.inf,np.inf),
    'wGj':(-1,1),
    'wIj':(-1,1),
    
    'aYij':(0,1),
    'Yij':(0,np.inf),
        }



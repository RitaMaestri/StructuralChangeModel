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
    'GDPPI':np.array([900]),
    'GDP':np.array([1000]),


 # must be greater than pL L

    'Cj':np.array([438,378]),
    'pCj':np.array([4,2]),  

    'Sj':np.array([438,378]),
    'pSj':np.array([4,3]),#nice
   
    'Kj':np.array([100,11]),
    'Lj':np.array([100,202]),

    'KLj':np.array([106,330]),
    'pKLj':np.array([3,7]),
    
    'Dj':np.array([438,378]), 
    'pDj':np.array([4,2]),

    'Xj':np.array([300,400]),


    'Yj':np.array([110,201]),
    'pYj':np.array([4,5]), 

    
    'Mj':np.array([408,308]),
    'pMj':np.array([4,2]),

       
    'Yij':np.array([[148,157],[202,265]]),
    }


parameters= {
    'L':np.array([111]),
    'K':np.array([111]),
    'wB':np.array([.1]),
    'GDPreal':np.array([2000]), 

    'pXj':np.array([10,10]),
    
    'pCtp':np.array([20,20]),
    'Ctp':np.array([300,400]),
    'alphaKj':np.array([.3,.7]),
    'alphaLj':np.array([.2,.8]),
    'aKLj':np.array([0.2,0.3]),
    'alphaCj':np.array([0.2,0.8]),
    'alphaXj':np.array([0.4,0.6]),
    'alphaDj':np.array([0.3,0.7]),
    'betaDj':np.array([0.8,0.2]),
    'betaMj':np.array([0.3,0.7]),
    'sigmaXj':np.array([0.9,0.5]),
    'sigmaSj':np.array([-.8,-0.8]),
    
    'aYij':np.array([[0.5,0.5],[0.5,0.5]])
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
    'pCtp':(0,np.inf),
    'Ctp':(0,np.inf),
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
    
    'aYij':(0,1),
    'Yij':(0,np.inf),
        }



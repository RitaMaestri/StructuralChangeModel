import numpy as np

#CAUTION: if you use the solver dict_minimize, 
#the bounds for the variables are (0,1) for variables whose initial guess is <1, (0,inf) for the rest
#if you use least_square it's (0,inf) in any case

variables = {
    'pL':np.array([1]),
    'bKL':np.array([.7]),
    'pK':np.array([4]),
    'wB':np.array([.5]),
    'B':np.array([100]),
    'GDPPI':np.array([900]),
    'GDP':np.array([900]),
    
 # must be greater than pL L

    'Cj':np.array([438,378]),
    'pCj':np.array([4,2]),  

    'Sj':np.array([438,378]),

   
    'Kj':np.array([100,11]),
    'Lj':np.array([100,202]),

    'KLj':np.array([106,330]),
    
    'Dj':np.array([438,378]), 
    'pDj':np.array([4,2]),

    'Xj':np.array([300,400]),
    'pXj':np.array([80,1]),


    'Yj':np.array([110,201]),
    'pYj':np.array([4,5]), 

    
    'Mj':np.array([408,308]),
    'pMj':np.array([4,2]),

       
    'Yij':np.array([[148,157],[202,265]]),
    }


parameters= {
    'L':np.array([111]),
    'K':np.array([1]),
    'GDPreal':np.array([2000]), # must be greater than pL L

    'pSj':np.array([4,3]),#nice
    'pKLj':np.array([3,7]),#

    
    'pCtp':np.array([1,1]),
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
    'sigmaSj':np.array([.8,0.8]),
    
    'aYij':np.array([[0.5,0.5],[0.5,0.5]])
    }
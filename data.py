import numpy as np

#CAUTION: if you use the solver dict_minimize, 
#the bounds for the variables are (0,1) for variables whose initial guess is <1, (0,inf) for the rest
#if you use least_square it's (0,inf) in any case

variables = {
    'GDPPI':np.array([900]),
    'GDP':np.array([900]),
    'bKL':np.array([0.8]),
    'pK':np.array([4]),
    
    'pKLj':np.array([3,7]),
    'pYj':np.array([4,5]),
    'pCj':np.array([4,2]),
    'KLj':np.array([106,330]),
    'Lj':np.array([100,202]),
    'Kj':np.array([100,11]),
    'Cj':np.array([438,378]),
    'Yj':np.array([110,201]),

    'Yij':np.array([[148,157],[202,265]]),
    }


parameters= {
    
    'pL':np.array([1]),
    'K':np.array([111]),
    'L':np.array([302]),
    'GDPreal':np.array([2000]),# must be greater than pL L
    'alphaKj':np.array([.5,.5]),
    'alphaLj':np.array([.5,.5]),
    'alphaCj':np.array([0.9,0.1]),
    'aKLj':np.array([0.5,0.5]),
    'pCtp':np.array([1,1]),
    'Ctp':np.array([1.5,1.6]),
    
    'aYij':np.array([[0.2,0.1],[0.4,0.5]])
    }
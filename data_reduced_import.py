import numpy as np

#CAUTION: if you use the solver dict_minimize, 
#the bounds for the variables are (0,1) for variables whose initial guess is <1, (0,inf) for the rest
#if you use least_square it's (0,inf) in any case

variables = {
    'pK':np.array([4]),
    'GDPPI':np.array([900]),
    'bKL':np.array([.7]),
    'GDP':np.array([900]),

    'KLj':np.array([106,330]),
    'Kj':np.array([100,11]),
    'pKLj':np.array([3,7]),
    'Lj':np.array([100,202]),
    'pCj':np.array([4,2]),
    'Cj':np.array([438,378]),
    'Yj':np.array([110,201]),
    'pYj':np.array([4,5]),
    'Mj':np.array([408,308]),
    'pSj':np.array([4,3]),
    'Sj':np.array([438,378]),

    'Yij':np.array([[148,157],[202,265]]),
    }


parameters= {
    'K':np.array([511]),
    'L':np.array([100]),
    'GDPreal':np.array([2000]), # must be greater than pL L
    'pL':np.array([1]),

    'pMj':np.array([4,2]),   
    'pCtp': np.array([1,1]),
    'Ctp': np.array([300,400]),
    'alphaKj': np.array([.3,.7]),
    'alphaLj': np.array([.8,.2]),
    'aKLj': np.array([0.2,0.3]),
    'alphaCj': np.array([0.8,0.2]),
    'betaYj': np.array([0.8,0.2]),
    'betaMj': np.array([0.3,0.7]),
    'sigmaXj': np.array([-0.5,0.5]),
    'sigmaSj': np.array([-0.2,0.8]),
    
    'aYij':np.array([[0.5,0.5],[0.5,0.5]])
    }

import numpy as np

#if you use minimize as a solver the bounds are (0,1) for the variables whose initial guess is <1, (0,inf) otherwise
variables= {
    'pL':np.array([1]),
    'pK':np.array([4]),
    'Lj':np.array([150,200]),
    'Kj':np.array([100,200]),
    'pKLj':np.array([3,7]),
    'alphaCj':np.array([0.9,0.1]),
    }

parameters= {
    
    'K':np.array([334]),
    'L':np.array([183]),
    'bKL':np.array([0.8]),
    'alphaKj':np.array([.5,.5]),
    'alphaLj':np.array([.5,.5]),
    'KLj':np.array([438,380]),
    }
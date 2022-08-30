import numpy as np
import import_csv as imp

#CAUTION: if you use the solver dict_minimize, 
#the bounds for the variables are (0,1) for variables whose initial guess is <1, (0,inf) for the rest
#if you use least_square it's (0,inf) in any case

variables = {
    'pK':np.array([1]),
    'L':np.array([sum(imp.pLLj)]),
    'GDP':np.array([10e+8]),# must be greater than pL L
    
    
    'pKLj':[1] * len(imp.pLLj),
    'bKLj':[.5] * len(imp.pLLj),
    'pCj':[1] * len(imp.pLLj),
    'Lj':imp.pLLj,
    'Kj':imp.pKKj,
    'Cj':imp.pCjCj,
    'Yj':imp.pYj,
    'alphaKj':[0.5] * len(imp.pLLj),
    'alphaLj':[0.5] * len(imp.pLLj),
    'alphaCj':[0.5] * len(imp.pLLj),
    'aKLj':[0.1] * len(imp.pLLj),
    
    
    'aYij':np.tile(np.array([0.5]),(len(imp.pLLj),len(imp.pLLj)) ),
    'Yij':imp.pYiYij*0.9
    }


parameters= {
    'pL':np.array([imp.pL]),
    'K':np.array([imp.K]),
    
    
    'KLj':imp.KLj,
    'pCj*Cj':imp.pCjCj,
    'pYj*Yj':imp.pYjYj,
    'pL*Lj':imp.pLLj,
    'pK*Kj':imp.pKKj,
    'pKLj*KLj':imp.pKLjKLj,
    'pYj':imp.pYj,
    
    
    'pYj*Yij':imp.pYiYij,
    }










# variables = {
#     'pK':np.array([4]),
#     'L':np.array([302]),
#     'GDP':np.array([4000]),# must be greater than pL L
    
    
#     'pKLj':np.array([3,7]),
#     'bKLj':np.array([1,1]), 
#     'pCj':np.array([4,2]),
#     'Lj':np.array([100,202]),
#     'Kj':np.array([100,11]),
#     'Cj':np.array([438,378]),
#     'Yj':np.array([110,201]),
#     'alphaKj':np.array([.5,.5]),
#     'alphaLj':np.array([.5,.5]),
#     'alphaCj':np.array([0.5,0.5]),
#     'aKLj':np.array([0.5,0.5]),
    
    
#     'aYij':np.array([[0.2,0.1],[0.4,0.5]]),
#     'Yij':np.array([[148,157],[202,265]]),
#     }


# parameters= {
#     'pL':np.array([1]),
#     'K':np.array([111]),
    
    
#     'KLj':np.array([1,1]),
#     'pCj*Cj':np.array([27,202]),
#     'pYj*Yj':np.array([12,52]),
#     'pL*Lj':np.array([157,145]),
#     'pK*Kj':np.array([141,89]),
#     'pKLj*KLj':np.array([101,99]),
#     'pYj':np.array([4,5]),
    
    
#     'pYj*Yij':np.array([[200,202], [100,50]]),
#     }
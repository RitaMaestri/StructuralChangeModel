import numpy as np


variables = {
    'L': np.array([2000]),
    'K': np.array([1000]),
    'B': np.array([-12400]),
    'R': np.array([986040]),
    'GDP': np.array([1912200]),
    'wB': np.array([-0.005]),
    
    'wGj¬E': np.array([0.2,]),
    'wIj¬E': np.array([0.2,]),

    'alphaKj': np.array([0.5, 0.5]),
    'alphaLj': np.array([0.5, 0.5]),
    'aKLj': np.array([0.5, 0.5]),
    'alphaCj': np.array([0.5, 0.5]),
    'alphaXj': np.array([0.5, 0.5]),
    'alphaDj': np.array([0.5, 0.5]),
    'betaDj': np.array([0.5, 0.5]),
    'betaMj': np.array([0.5, 0.5]),
    
    'pSj': np.array([100, 50]),
    'pMj': np.array([100, 100]),
    'pCj': np.array([100, 50]),


    'Sj': np.array([10000, 1000]),
    'Cj': np.array([1000, 1000]),
    'Kj': np.array([1000, 5000]),
    'Mj': np.array([4000, 1000]),
    'Lj': np.array([1000, 1000]),
    'KLj': np.array([1000, 1000]),
    'Xj': np.array([1000, 1000]),
    'Yj': np.array([1000, 1000]),
    'Gj¬E': np.array([1000,]),
    'Ij¬E': np.array([1000,]),
    'Dj': np.array([1000, 1000]),


    'Yij': np.array([[1000, 1000], [1000, 1000]]),
    'aYij': np.array([[0.5, 0.5], [0.5, 0.5]]),
}


##########################   DATA TAKEN FORM TES 17 KLEM   ########################



parameters = {
    
    'G_E': np.array([0]),
    'I_E': np.array([0]),
    
    'wG_E': np.array([0]),
    'wI_E': np.array([0]),

    'pK': np.array([100]),
    'pL': np.array([100]),
    'bKL': np.array([1]),


    'pKLj': np.array([100, 100]),
    'pYj': np.array([100, 100]),
    'pXj': np.array([100, 100]),
    'pDj': np.array([100, 100]),

    'sigmaXj': np.array([-2, -2]),
    'sigmaSj': np.array([2, 3.761]),
    'pXj*Xj': np.array([653876, 18682]),
    'pMj*Mj': np.array([630908, 54103]),
    'pCj*Cj': np.array([931062.92, 54977.09]),
    'pCj*Gj': np.array([500066.20, 0]),
    'pCj*Ij': np.array([438623.81, 0]),
    'pL*Lj': np.array([1116558, 24864]),
    'pK*Kj': np.array([736706, 34150]),
    'pSj*Yij': np.array([[1741427, 38409], [77866, 27313]]),
}



bounds = {
    'pL': (0, np.inf),
    'pK': (0, np.inf),
    'L': (0, np.inf),
    'K': (0, np.inf),
    'B': (-np.inf, np.inf),
    'wB': (-1, 1),
    'GDP': (-np.inf, np.inf),
    'bKL': (0, np.inf),
    'R': (0, np.inf),

    'Cj': (0, np.inf),
    'pCj': (0, np.inf),
    'Sj': (0, np.inf),
    'pSj': (0, np.inf),
    'Kj': (0, np.inf),
    'Lj': (0, np.inf),
    'KLj': (0, np.inf),
    'pKLj': (0, np.inf),
    'Dj': (0, np.inf),
    'pDj': (0, np.inf),
    'Xj': (0, np.inf),
    'pXj': (0, np.inf),
    'Yj': (0, np.inf),
    'pYj': (0, np.inf),
    'Mj': (0, np.inf),
    'pMj': (0, np.inf),
    'G_E': (0, np.inf),
    'I_E': (0, np.inf),
    'Gj¬E': (0, np.inf),
    'Ij¬E': (0, np.inf),
    'alphaKj': (0, 1),
    'alphaLj': (0, 1),
    'aKLj': (0, 1),
    'alphaCj': (0, np.inf),
    'alphaXj': (-np.inf, np.inf),
    'alphaDj': (-np.inf, np.inf),
    'betaDj': (-np.inf, np.inf),
    'betaMj': (-np.inf, np.inf),
    'sigmaXj': (-np.inf, np.inf),
    'sigmaSj': (-np.inf, np.inf),
    'wG_E': (-1, 1),
    'wI_E': (-1, 1),
    'wGj¬E': (-1, 1),
    'wIj¬E': (-1, 1),
    'pCj*Cj': (0, np.inf),
    'pCj*Gj': (0, np.inf),
    'pCj*Ij': (0, np.inf),
    'pL*Lj': (0, np.inf),
    'pK*Kj': (0, np.inf),
    'pYj*Yj': (0, np.inf),
    'pXj*Xj': (0, np.inf),
    'pMj*Mj': (0, np.inf),

    'pSj*Yij': (0, np.inf),
    'aYij': (0, 1),
    'Yij': (0, np.inf),
}



##### THE FOLLOWING VARIABLES AND PARAMETERS ARE AT AN EQUILIBRIUM POINT######

# variables = {
# 'L': np.array([111]),
# 'K': np.array([111]),
# 'B': np.array([400.]),
# 'R': np.array([2280.]),
# 'GDP': np.array([996.88441179]),
# 'wB': np.array([0.1]),


# 'alphaKj': np.array([0.2, 0.6, 0.3]),
# 'alphaLj': np.array([0.2, 0.7, 0.1]),
# 'aKLj': np.array([0.2, 0.3, 0.4]),
# 'alphaCj': np.array([0.2, 0.3, 0.5]),
# 'alphaXj': np.array([0.2, 0.5, 0.3]),
# 'alphaDj': np.array([0.2, 0.2, 0.6]),
# 'betaDj': np.array([0.1, 0.2, 0.7]),
# 'betaMj': np.array([0.3, 0.3, 0.4]),
# 'wGj': np.array([0.05, 0.05, 0.05]),
# 'wIj': np.array([0.06, 0.06, 0.06]),


# 'pSj': np.array([ 17.15621324,  17.27179156, 113.41576942]),
# 'pCj': np.array([ 17.15621324,  17.27179156, 113.41576942]),
# 'pMj': np.array([10., 10., 15.]),


# 'Sj': np.array([366.49353218, 311.05427397, 511.38893855]),
# 'Cj': np.array([26.57929192, 39.60214536, 10.05151229]),
# 'Kj': np.array([9.57256762e-02, 4.00080396e-01, 1.10504194e+02]),
# 'Mj': np.array([623.49383332, 526.34349351, 210.97820322]),
# 'Lj': np.array([  0.28412676,   1.38540819, 109.33046505]),
# 'KLj': np.array([ 20.97073707,  31.26787511, 282.87419296]),
# 'Dj': np.array([121.77372871, 112.51488385, 584.47580507]),
# 'Xj': np.array([ 25.3334232 ,  97.6421741 , 922.21935624]),
# 'Yj': np.array([104.85368536, 104.22625036, 707.18548241]),
# 'Gj': np.array([11.65758418, 11.57957467,  1.76342321]),
# 'Ij': np.array([13.98910101, 13.8954896 ,  2.11610785]),


# 'Yij': np.array([[ 20.97073707,  10.42262504, 282.87419296],
#        [ 52.42684268,  52.11312518, 141.43709648],
#        [ 31.45610561,  41.69050014, 424.31128945]]),
# 'aYij': np.array([[0.2, 0.1, 0.4],
#        [0.5, 0.5, 0.2],
#        [0.3, 0.4, 0.6]])
# }


# parameters = {

#     'pK': np.array([0.55941912]),
#     'pL': np.array([0.18847494]),
#     'bKL': np.array([43.12266376]),
    
    
#     'pXj': np.array([10, 10, 15]),
#     'pKLj': np.array([0.01276798, 0.01192985, 0.72845292]),
#     'pYj': np.array([46.09442285, 55.72140383, 78.65768643]),
#     'pDj': np.array([ 0.43279332,  0.96902423, 93.81893384]),


#     'sigmaXj': np.array([0.5, 0.1, 0.4]),
#     'sigmaSj': np.array([-0.8, -0.8, -0.8]),
#     'pXj*Xj': np.array([  253.33423201,   976.42174097, 13833.29034354]),
#     'pMj*Mj': np.array([6234.9383332 , 5263.43493507, 3164.67304825]),
#     'pCj*Gj': np.array([200., 200., 200.]),
#     'pCj*Cj': np.array([ 456.,  684., 1140.]),
#     'pCj*Ij': np.array([240., 240., 240.]),
#     'pYj*Yj':np.array([ 4833.17011034,  5807.63298561, 55625.57392036]),
#     'pL*Lj': np.array([ 0.05355077,  0.26111473, 20.60605305]),
#     'pK*Kj': np.array([5.35507738e-02, 2.23812624e-01, 6.18181591e+01]),
#     'pSj*Yij': np.array([[  359.77843702,   178.81277764,  4853.04997472],
#            [  905.50549897,   900.08703569,  2442.87204942],
#            [ 3567.61842048,  4728.36015124, 48123.59136575]])

# }









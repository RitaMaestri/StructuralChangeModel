from scipy import optimize
import numpy as np
import sys
from math import sqrt

sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq


def to_dict(vec, dvec):  #takes array, returns dict of arrays (of equal dimensions and keys as dvec)
    lengths = np.array([sum(np.shape(item)) for item in dvec.values()])
    N=int(sqrt(max(lengths)))
    keys=dvec.keys()
    #create array of arrays
    vec = np.split(vec,np.cumsum(lengths))[:-1]
    #create array of arrays and matrices (if needed)
    vec = [np.reshape(i, (N, N)) if len(i) == N**2 else i for i in vec]
    return dict(zip(keys, vec))


def to_array(dvar):  #takes dict returns array
    var=[np.array(dvar[k]) for k in dvar.keys()]
    return np.array(np.hstack([var[i].flatten() for i in range(len(var))]))


def system(var, par):
    
    d = {**var, **par}
    
    return np.hstack([
        eq.eqKLj(d['KLj'], d['bKLj'], d['Lj'], d['Kj'], d['alphaLj'], d['alphaKj']),
        eq.eqFj(d['Lj'],d['GDP'],d['pK'],d['K'],d['L'],d['KLj'],d['pKLj'],d['alphaLj']),
        eq.eqFj(d['Kj'],d['GDP'],d['pL'],d['L'],d['K'],d['KLj'],d['pKLj'],d['alphaKj']),
        eq.eqYij(d['Yij'],d['aYij'],d['Yj']),
        eq.eqKL(d['KLj'],d['aKLj'],d['Yj']),
        eq.eqpYj(d['pYj'],d['aKLj'],d['pKLj'],d['aYij']),
        eq.eqCj(d['Cj'],d['alphaCj'],d['pCj'],d['GDP']),
        eq.eqYj(d['Yj'],d['Cj'],d['Yij']),
        eq.eqF(d['L'],d['Lj']),
        eq.eqF(d['K'],d['Kj']),
        eq.eqpCj(d['pCj'],d['pYj'])
        ])


def dict_least_squares(f, dvar, dpar):
    result = optimize.least_squares(
        lambda x,y: f(to_dict(x,dvar), to_dict(y,dpar)), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= list([to_array(dpar)])
    )
    return result;


def dict_fsolve(f, dvar, dpar):
    result = optimize.fsolve(
        lambda x,y: f(to_dict(x,dvar), y), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= dpar
    )
    return result;


variables = {'pL': np.array([59057.25132478]),
  'pK': np.array([-38341.90453179]),
  'KLj': np.array([ 0.36712262, 14.05476505]),
  'Lj': np.array([-0.46674462,  3.50519154]),
  'Kj': np.array([ 1.12095355, -0.21571321]),
  'pKLj': np.array([-18116.96037197,   1565.67981374]),
  'Yj': np.array([1.30167461, 7.02791538]),
  'pYj': np.array([1361.56045735, 8095.77017803]),
  'Cj': np.array([-0.84879547,  0.0140812 ]),
  'pCj': np.array([1361.56046012, 8095.77017959]),
  'aYij': np.array([[0.76367541, 0.28395635],
        [2.27757584, 0.56545415]])}


# variables = { 
#     'pL':np.array([1]),
#     'pK':np.array([2]),
#     'KLj':np.array([80,80]),
#     'Lj':np.array([1,1]),
#     'Kj':np.array([3,15]),
#     'pKLj':np.array([15,15]),
#     'Yj':np.array([1,2]),
#     'pYj':np.array([1,4]),
#     'Cj':np.array([2,1]),
#     'pCj':np.array([1,3]),
#     'aYij':np.array([[1,0],[1,1]])
#     }
    
parameters= {
    'GDP':np.array([1]),
    'K':np.array([2]),
    'L':np.array([3]),
    'bKLj':np.array([1,2]),
    'alphaKj':np.array([1,2]),
    'alphaLj':np.array([3,4]),
    'alphaCj':np.array([1,2]),
    'aKLj':np.array([1,2]),
    'Yij':np.array([[1,2],[3,4]])
    }


#add the lengths of variables to the parameters

solLS= dict_least_squares(system, variables, parameters).x
solFS= dict_fsolve(system, variables, parameters)

print(system(to_dict(solLS,variables), parameters))
#print(system(to_dict(solFS,variables), parameters))

#test radom equations

d = {**to_dict(solLS, variables), **parameters}
#print(eq.eqCj(d["Cj"],d["alphaCj"],d["pCj"],d["GDP"]))



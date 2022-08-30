#SOLVEURS ALTERNATIVES

from scipy import optimize
import numpy as np
from math import sqrt


# TYPE CONVERTERS 

def to_dict(vec, dvec):  #takes array, returns dict of arrays (of equal dimensions and keys as dvec)
    lengths = np.array([np.product(np.shape(item)) for item in dvec.values()])
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



########################  FSOLVE  ########################
def dict_fsolve(f, dvar, dpar):
    result = optimize.fsolve(
        lambda x,y: f(to_dict(x,dvar), y), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= dpar
    )
    result.dvar= to_dict(result, dvar)
    result.d= {**result.dvar, **dpar}
    return result;



########################  MINIMIZE  ########################
def cost_function(array):
    return np.mean(np.square(array))


def bounds(array_var):
    bounds=list()
    for variables in array_var:
        if variables < 1 :
            bounds.append((0, 1))
        else:
            bounds.append((10e-16, np.inf))
    return bounds

#the bounds are (0,1) for variables whose initial guess is <1, (0,inf) for the rest

def dict_minimize(f, dvar, dpar):
    result = optimize.minimize(
        fun=lambda x,y: cost_function(f(to_dict(x,dvar), to_dict(y,dpar))),# wrap the argument in a dict
        x0=to_array(dvar), # unwrap the initial dictionary
        bounds=bounds(to_array(dvar)),
        args= to_array(dpar),
        method="SLSQP",
        options=dict(
            maxiter=1000000,
            iprint=2,
        )
        #verbose=2,
    )
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result;



####################  LEAST_SQUARE  #########################

def dict_least_squares(f, dvar, dpar):
    result = optimize.least_squares(
        lambda x,y: f(to_dict(x,dvar), to_dict(y,dpar)),# wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        bounds=(0,np.inf),
        args= list([to_array(dpar)],),
    )
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result;
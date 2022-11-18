#SOLVEURS ALTERNATIVES

from scipy import optimize
import numpy as np
from math import sqrt


# TYPE CONVERTERS 

def to_dict(vec, dvec):  #takes array, returns dict of arrays (of equal dimensions and keys as dvec)
    lengths = np.array([np.product(np.shape(item)) for item in dvec.values()])
    N=int(min(lengths[lengths!=1]))
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
    return sqrt(sum(np.square(array)))



def dict_minimize(f, dvar, dpar, bounds):
    result = optimize.minimize(
        fun=lambda x,y: cost_function(f(to_dict(x,dvar), to_dict(y,dpar))),# wrap the argument in a dict
        x0=to_array(dvar), # unwrap the initial dictionary
        bounds=bounds,
        args= to_array(dpar),
        method='TNC',
        #options=dict(
        #    maxiter=1000000,
        #    iprint=2,
        #)
        verbose=2,
    )
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result;


####################  LEAST_SQUARE  #########################

def dict_least_squares(f, dvar, dpar,bounds, verb=1):
    result = optimize.least_squares(
        lambda x,y: f(to_dict(x,dvar), to_dict(y,dpar)),# wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        bounds=bounds,
        args= list([to_array(dpar)],),
        verbose=verb
    )
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result;

##########   BASINHOPPING   ###############


class MyBounds:
    def __init__(self, bounds ):
        self.xmin,self.xmax= [[row[i] for row in bounds] for i in (0,1)]
        self.bounds = [np.array([row[0]+1e-14,row[1]]) for row in bounds]
 
        self.epsilon = 1e-14
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x < self.xmax))
        tmin = bool(np.all(x > self.xmin))
        return tmax and tmin
    

def dict_basinhopping(f, dvar, dpar, mybounds):
    result = optimize.basinhopping(
        lambda x,y: cost_function(f(to_dict(x,dvar), to_dict(y,dpar))),# wrap the argument in a dict
        x0=to_array(dvar), # unwrap the initial dictionary     
        #niter=2,
        minimizer_kwargs = dict(method="L-BFGS-B",  bounds=mybounds.bounds, args= to_array(dpar)),
        accept_test=mybounds,
        stepsize=10
    )
    
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result



##########   SHGO NOT WORKONG   ###############


def dict_shgo(f, dvar, dpar,bounds): # unwrap the initial dictionary
    result = optimize.shgo(
        lambda x,y: cost_function(f(to_dict(x,dvar), to_dict(y,dpar))),# wrap the argument in a dict
        bounds=bounds,
        args= list([to_array(dpar)],),
        minimizer_kwargs = dict(method="L-BFGS-B"),
    )
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result;

from scipy import optimize
import numpy as np
import sys
from math import sqrt


sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from data_reduced import variables, parameters



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



#system of equations
def system_reduced(var, par):
    
    d = {**var, **par}

    return np.hstack([
        eq.eqKLj(KLj =d['KLj'], bKL=d['bKL'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        eq.eqCj(Cj=d['KLj'],alphaCj=d['alphaCj'],pCj=d['pKLj'],pL=d['pL'],Lj=d['Lj'],pK=d['pK'],Kj=d['Kj']),
        eq.eqF(F=d['L'],Fj=d['Lj']),
        eq.eqF(F=d['K'],Fj=d['Kj']),
        #eq.eqpCj(pCj=d['pCj'],pYj=d['pKLj']),
        #eq.eqpCj(pCj=d['Cj'],pYj=d['KLj']),
        #eq.eqGDP(d['GDP'], d['pL'], d['L'], d['pK'], d['K'])
        ])


def cost_function(array):
    return np.mean(np.square(array))
    



#least square solver that takes as an argument a dictionary instead of an array
def dict_least_squares(f, dvar, dpar):
    result = optimize.least_squares(
        lambda x,y: f(to_dict(x,dvar), to_dict(y,dpar)),# wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        bounds=(0,np.inf),
        args= list([to_array(dpar)],),
        gtol=10e-14,
        ftol=10e-14,
        xtol=10e-14,
        #verbose=2,
        x_scale=np.concatenate(list({#same order of magnitude as initial guess
            'pK':np.array([1]),
            'Lj':np.array([100,100]),
            'Kj':np.array([100,100]),
            'pKLj':np.array([1,1]),
            'alphaCj':np.array([0.1,0.1]),
            }.values()))
    )
    return result;


def dict_minimize(f, dvar, dpar):
    result = optimize.minimize(
        lambda x,y: cost_function(f(to_dict(x,dvar), to_dict(y,dpar))),# wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        bounds=((0, None),(0, None),(0, None),(0, None),(0, None),(0, None), (0, None),(0, None), (0,None),(0,None)),
        args= to_array(dpar),
        method="SLSQP"
        #verbose=2,
    )
    return result;



#fsolve solver that takes as an argument a dictionary instead of an array
def dict_fsolve(f, dvar, dpar):
    result = optimize.fsolve(
        lambda x,y: f(to_dict(x,dvar), y), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= dpar
    )
    return result;



#solver
sol= dict_minimize(system_reduced, variables, parameters)

#sol= dict_fsolve(system, variables, parameters)


print("\n system of equations calculated at the solution found by minimize (an array of zeros is expected): \n \n", system_reduced(to_dict(sol.x,variables), parameters))


print("\n solution attributes: \n \n",sol)





# #for checks
d = {**to_dict(sol.x, variables), **parameters}

# #takes the array jacobian, tirns it into a dictionary making explicit the function and derivation variables
# def to_dict_jac(jac, dvec):
#     keys=list(["eqKLj", "eq.eqLj","eq.eqKj", "eq.eqCj","eq.eqL","eq.eqK","eq.eqpCj","eq.eqCjKLj"])
#     #vec=int(vec)
#     jac = dict(zip(keys, jac))
#     for i in keys:
#         jac[i]=to_dict(jac[i],dvec)
#     return jac

# jacobian=to_dict_jac(sol.jac,variables)


 




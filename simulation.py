from scipy import optimize
import numpy as np
import sys
from math import sqrt
from data import variables, parameters

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
        eq.eqKLj(KLj =d['KLj'], bKL=d['bKL'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj']),
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']),
        eq.eqpYj(pYj=d['pYj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij']),
        eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],pL=d['pL'],Lj=d['Lj'],pK=d['pK'],Kj=d['Kj']),
        eq.eqYj(Yj=d['Yj'],Cj=d['Cj'],Yij=d['Yij']),
        eq.eqF(F=d['L'],Fj=d['Lj']),
        eq.eqF(F=d['K'],Fj=d['Kj']),
        eq.eqpCj(pCj=d['pCj'],pYj=d['pYj']),
        #eq.eqGDP(d['GDP'], d['pL'], d['L'], d['pK'], d['K'])
        ])


#least square function that takes as an argument a dictionary instead of an array
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
        #x_scale=
    )
    return result;


def dict_fsolve(f, dvar, dpar):
    result = optimize.fsolve(
        lambda x,y: f(to_dict(x,dvar), y), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= dpar
    )
    return result;




sol= dict_least_squares(system, variables, parameters)


solLS=sol.x
#solFS= dict_fsolve(system, variables, parameters)

d = {**to_dict(solLS, variables), **parameters}


#takes the array jacobian, tirns it into a dictionary making explicit the function and derivation variables
def to_dict_jac(jac, dvec):  
    keys=list(["eqKLj", "eq.eqLj","eq.eqKj", "eq.eqYij","eq.eqKL","eq.eqpYj","eq.eqCj","eq.eqYj","eq.eqL","eq.eqK","eq.eqpCj"])
    #vec=int(vec)
    jac = dict(zip(keys, jac))
    for i in keys:
        jac[i]=to_dict(jac[i],dvec)
    return jac

jacobian=to_dict_jac(sol.jac,variables)

print(system(to_dict(solLS,variables), parameters))
#print(system(to_dict(solFS,variables), parameters))


print(to_dict(solLS,variables))

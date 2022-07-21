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


def to_dict_jac(vec, dvec):  #takes array, returns dict of arrays (of equal dimensions and keys as dvec)
    keys=dvec.keys()
    vec=int(vec)
    return dict(zip(keys, vec))

def to_array(dvar):  #takes dict returns array
    var=[np.array(dvar[k]) for k in dvar.keys()]
    return np.array(np.hstack([var[i].flatten() for i in range(len(var))]))


def system(var, par):
    

    d = {**var, **par}
    

    return np.hstack([
        #eq.eqKLj(d['KLj'], d['bKL'], d['Lj'], d['Kj'], d['alphaLj'], d['alphaKj']),
        eq.eqKLj(KLj=d['Cj'], bKL=d['bKL'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['Cj'],pKLj=d['pKLj'],alphaFj=d['alphaLj']),
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['Cj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),
        # eq.eqFj(d['Lj'],d['pL'],d['KLj'],d['pKLj'],d['alphaLj']),
        # eq.eqFj(d['Kj'],d['pK'],d['KLj'],d['pKLj'],d['alphaKj']),
        
        #eq.eqYij(d['Yij'],d['aYij'],d['Yj']),
        #eq.eqKL(d['KLj'],d['aKLj'],d['Yj']),
        #eq.eqpYj(d['pYj'],d['aKLj'],d['pKLj'],d['aYij']),
        eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],pL=d['pL'],L=d['L'],pK=d['pK'],K=d['K']),
        #eq.eqYj(d['Yj'],d['Cj'],d['Yij']),
        eq.eqF(F=d['L'],Fj=d['Lj']),
        eq.eqF(F=d['K'],Fj=d['Kj']),
        eq.eqpCj(pCj=d['pCj'],pYj=d['pKLj']),
       # eq.eqpCj(d['pCj'],d['pYj']),
        #eq.eqGDP(d['GDP'], d['pL'], d['L'], d['pK'], d['K'])
        ])


def dict_least_squares(f, dvar, dpar):
    result = optimize.least_squares(
        lambda x,y: f(to_dict(x,dvar), to_dict(y,dpar)),# wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        bounds=(0,[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,1,np.inf,np.inf,]),
        args= list([to_array(dpar)],),
        ftol=None,
        xtol=None,
        gtol=10e-15,
        #verbose=2,
        x_scale=np.array([1,100,100,100,100,100,100,0.1,0.1,1,1]),
    )
    return result;


def dict_fsolve(f, dvar, dpar):
    result = optimize.fsolve(
        lambda x,y: f(to_dict(x,dvar), y), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= dpar
    )
    return result;



variables = {
    'pK':np.array([.1]),
    'Lj':np.array([200,20000]),
    'Kj':np.array([500,20000]),
    'pKLj':np.array([1,1]),
    'alphaCj':np.array([0.9,0.1]),
    'pCj':np.array([4,20]),
    }


parameters= {
    'pL':np.array([1]),
    'K':np.array([334]),
    'L':np.array([183]),
    'bKL':np.array([0.8]),
    'alphaKj':np.array([.5,.5]),
    'alphaLj':np.array([.5,.5]),
    'Cj':np.array([438,293]),
    }



# variables = {
#     'pK':np.array([4]),
#     'KLj':np.array([10,30]),
#     'Lj':np.array([200,2000]),
#     'Kj':np.array([4000,11]),
#     'pKLj':np.array([345,789]),
#     'Yj':np.array([11,12]),
#     'pYj':np.array([4,5]),
#     'alphaCj':np.array([0.9,0.1]),
#     'pCj':np.array([4,20]),
#     'aYij':np.array([[0.2,0.1],[0.4,0.5]])
#     }
    

# parameters= {
#     'pL':np.array([1]),
#     'K':np.array([334]),
#     'L':np.array([183]),
#     'bKL':np.array([0.8]),
#     'alphaKj':np.array([.5,.5]),
#     'alphaLj':np.array([.5,.5]),
#     'Cj':np.array([438.63792689187,78.3620731081594]),
#     'aKLj':np.array([0.5,0.5]),
#     'Yij':np.array([[148.892198751339,57.469874356791],[202.669132341745,65.9687945500961]])
#     }



sol= dict_least_squares(system, variables, parameters)
solLS=sol.x
#solFS= dict_fsolve(system, variables, parameters)

print(system(to_dict(solLS,variables), parameters))
#print(system(to_dict(solFS,variables), parameters))


d = {**to_dict(solLS, variables), **parameters}



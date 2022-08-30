import numpy as np
import sys
from data import variables, parameters

sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve


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
        eq.eqID(x=d['pCj'],y=d['pYj']),
        eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj']),
        eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        #eq.eqID(x=d['pCtp'],y=d['pCj']),
        #eq.eqID(x=d['Ctp'],y=d['Cj']),
        ])




sol= dict_least_squares(system, variables, parameters)


#solFS= dict_fsolve(system, variables, parameters)

print("\n \n solution: \n \n",sol.dvar)
print("\n \n parameters: \n \n",parameters)

print("\n system of equations calculated at the solution found by least_square (an array of zeros is expected): \n \n", system(sol.dvar, parameters))


d = {**sol.dvar, **parameters}


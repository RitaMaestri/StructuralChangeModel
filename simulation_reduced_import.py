import numpy as np
import sys
from data_reduced_import import variables, parameters

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
        eq.eqpYj(pYj=d['pYj'],pSj=d['pSj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij']),
        #eq.eqCET(Zj=d['Sj'], alphaXj=d['betaYj'],alphaYj=d['betaMj'],Xj=d['Yj'],Yj=d['Mj'],sigmaj=d['sigmaSj']),
        eq.eqCESquantity(Xj=d['Yj'], Zj=d['Sj'], alphaXj=d['betaYj'], alphaYj=d['betaMj'], pXj=d['pYj'], pYj=d['pMj'], sigmaj=d['sigmaSj']),
        eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaYj'], pXj=d['pMj'], pYj=d['pYj'], sigmaj=d['sigmaSj']),
        eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pYj'],alphaXj=d['betaMj'],alphaYj=d['betaYj'],sigmaj=d['sigmaSj']),
        eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],pL=d['pL'],Lj=d['Lj'],pK=d['pK'],Kj=d['Kj']),
        eq.eqSj(Sj=d['Sj'],Cj=d['Cj'],Yij=d['Yij']),
        eq.eqF(F=d['L'],Fj=d['Lj']),
        eq.eqF(F=d['K'],Fj=d['Kj']),
        eq.eqID(x=d['pCj'],y=d['pSj']),
        eq.eqGDPreduced(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],pMj=d['pMj'],Mj=d['Mj']),
        eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        ])




sol= dict_least_squares(system, variables, parameters)


#solFS= dict_fsolve(system, variables, parameters)

print("\n \n solution: \n \n",sol.dvar)
print("\n \n parameters: \n \n",parameters)

print("\n system of equations calculated at the solution found by least_square (an array of zeros is expected): \n \n", system(sol.dvar, parameters))
print(sum(abs( system(sol.dvar, parameters))))



d = {**sol.dvar, **parameters}
import numpy as np
import sys
from data import variables, parameters

sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve

for item in variables.keys(  ):
    if item in parameters:
        print("the same variable is both in variables and parameters")
        sys.exit()


def system(var, par):

    d = {**var, **par}

    return np.hstack([
        eq.eqKLj(KLj =d['KLj'], bKL=d['bKL'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
        
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
        
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj']),
        
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']),
        
        eq.eqpYj(pYj=d['pYj'],pSj=d['pSj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij']),
        
        eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj']),
        
        #eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj']),
        
        eq.eqCET(Zj=d['Yj'], alphaXj=d['alphaXj'],alphaYj=d['alphaDj'],Xj=d['Xj'],Yj=d['Dj'],sigmaj=d['sigmaXj']),
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj']),
        
        eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj']),
        
        eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj']),
        
        eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        eq.eqw(B=d['B'],wB=d['wB'],GDP=d['GDP']),
        
        eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],pL=d['pL'],Lj=d['Lj'],pK=d['pK'],Kj=d['Kj']),
        
        eq.eqSj(Sj=d['Sj'],Cj=d['Cj'],Yij=d['Yij']),
        
        eq.eqF(F=d['L'],Fj=d['Lj']),
        
        eq.eqF(F=d['K'],Fj=d['Kj']),
        
        eq.eqID(x=d['pCj'],y=d['pSj']),
        
        eq.eqID(x=d['pXj'],y=d['pMj']),
        
        eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        
        ])

len_var=len(np.hstack([x.flatten() for x in list(variables.values())]))
len_sys=len(system(variables,parameters))

if len_var != len_sys:
    print("system has ",len_sys, " equations")
    print("there are ",len_var, " variables")
    sys.exit()


sol= dict_least_squares(system, variables, parameters)

#sol= dict_minimize(system, variables, parameters)


#solFS= dict_fsolve(system, variables, parameters)

#print("\n \n solution: \n \n",sol.dvar)
#print("\n \n parameters: \n \n",parameters)

print("\n system of equations calculated at the solution found by least_square (an array of zeros is expected): \n \n", system(sol.dvar, parameters))
print(sum(abs( system(sol.dvar, parameters))))



d = {**sol.dvar, **parameters}


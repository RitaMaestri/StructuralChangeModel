import numpy as np
import sys
from data_calibration import variables, parameters

sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve


def system(var, par):

    d = {**var, **par}
#make an error if the variables are counted twice
    
    return np.hstack([
        eq.eqF(F=d['L'],Fj=d['Lj']),
        eq.eqF(F=d['K'],Fj=d['Kj']),
        eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj']),
        
        eq.eqKLj(KLj =d['KLj'], bKLj=d['bKLj'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']),
        #eq.eqpYj(pYj=d['pYj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij']),
        eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],pL=d['pL'],Lj=d['Lj'],pK=d['pK'],Kj=d['Kj']),
        #eq.eqYj(Yj=d['Yj'],Cj=d['Cj'],Yij=d['Yij']),
        eq.eqID(x=d['pCj'],y=d['pYj']),
        eq.eqCalibi(pX=d['pCj'], Xj=d['Cj'], data=d['pCj*Cj']),
        eq.eqCalibi(pX=d['pYj'], Xj=d['Yj'], data=d['pYj*Yj']),
        eq.eqCalibi(pX=d['pL'], Xj=d['Lj'], data=d['pL*Lj']),
        eq.eqCalibi(pX=d['pK'], Xj=d['Kj'], data=d['pK*Kj']),
        eq.eqCalibi(pX=d['pKLj'], Xj=d['KLj'], data=d['pKLj*KLj']),
        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj']),
        eq.eqCalibij(pYi=d['pYj'], Yij=d['Yij'], data=d['pYj*Yij']),
        ])
        


sol= dict_least_squares(system, variables, parameters)


#solFS= dict_fsolve(system, variables, parameters)

print("\n \n solution: \n \n",sol.dvar)
print("\n \n parameters: \n \n",parameters)

print("\n system of equations calculated at the solution found by least_square (an array of zeros is expected): \n \n", system(sol.dvar, parameters))

print("\n sum of residuals \n \n", sum(abs(system(sol.dvar, parameters))))

d = {**sol.dvar, **parameters}


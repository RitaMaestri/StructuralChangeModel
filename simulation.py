import numpy as np
import pandas as pd
import sys
import simple_calibration as cal
from data import variables, parameters, bounds, N
from import_csv import non_zero_index_G, non_zero_index_I, non_zero_index_C, non_zero_index_X, non_zero_index_M, non_zero_index_Yij
import import_csv as imp
sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve, dict_basinhopping, MyBounds

for k in (pavars := {**parameters,**variables}).keys():
    if not isinstance(pavars[k], np.ndarray):
        print(k, " is not an array!")
        sys.exit()


for item in variables.keys():
    if item in parameters:
        print("the same variable is both in variables and parameters")
        sys.exit()

for k in parameters.keys():
    for par in parameters[k].flatten():
        if par < bounds[k][0] or par > bounds[k][1]:
            print(par)
            print(bounds[k][0])
            print("parameter ", k ," out of bounds")
            sys.exit()
            
for k in variables.keys():
    for par in variables[k].flatten():
        if par < bounds[k][0] or par > bounds[k][1]:
            print("variable ", k ," out of bounds")
            sys.exit()

def augment_dict(d, key0, indexes, value_key):
    array=np.zeros(N)
    array[indexes] = d[value_key]
    d[key0] = array
    
def augment_dict2D(d, key0, indexes, value_key):
    array=np.zeros([N,N])
    for i,j,k in zip(indexes[0], indexes[1], range(len(indexes[0]))):
            array[i,j]=d[value_key][k]
    d[key0] = array

    
def system(var, par):

    d = {**var, **par}

    augment_dict(d, 'Gj', non_zero_index_G,'Gjn0')
    augment_dict(d, 'Ij', non_zero_index_I, 'Ijn0')
    augment_dict(d, 'Cj', non_zero_index_C, 'Cjn0')
    augment_dict(d, 'Xj', non_zero_index_X, 'Xjn0')
    augment_dict(d, 'Mj', non_zero_index_M, 'Mjn0')

    augment_dict2D(d, 'Yij', non_zero_index_Yij, 'Yijn0')

    return np.hstack([
        eq.eqKLj(KLj =d['KLj'],bKL=d['bKL'], bKLj=d['bKLj'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
        
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
        
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj'], _index=non_zero_index_Yij),
        
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']),
        
        eq.eqpYj(pYj=d['pYj'],pSj=d['pSj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], tauYj=d['tauYj']),
        
        #eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], _index=non_zero_index_X),#e-5
        
        #eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj']),
        
        #eq.eqCESprice(pZj=d['pYj'],pXj=d['pXj'],pYj=d['pDj'],alphaXj=d['alphaXj'],alphaYj=d['alphaDj'],sigmaj=d['sigmaXj']),
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj']),
        
        eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'],_index=non_zero_index_M),
        
        #eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj']),
        
        eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        #eq.eqwB(X=d['B'],wX=d['wB'],GDP=d['GDP']),
               
        eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],R=d['R'], _index=non_zero_index_C),
        
        eq.eqw(pXj=d['pCj'],Xj=d['Gj'],wXj=d['wGj'],GDP=d['GDP'], _index=non_zero_index_G),
        
        eq.eqw(pXj=d['pCj'],Xj=d['Ij'],wXj=d['wIj'],GDP=d['GDP'], _index=non_zero_index_I),
        
        #eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
        
        eq.eqF(F=d['L'],Fj=d['Lj']),
        
        eq.eqF(F=d['K'],Fj=d['Kj']),
        
        #eq.eqID(x=d['pCj'],y=d['pSj']),
        
        eq.eqID(x=d['pXj'],y=d['pMj']),
        
        eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pXj=d['pXj'], pCtp= d['pCtp'], pXtp=d['pXtp'], Cj= d['Cj'], Gj= d['Gj'], Ij= d['Ij'], Xj=d['Xj'], Mj=d['Mj'], Ctp= d['Ctp'], Gtp= d['Gtp'], Itp= d['Itp'], Xtp=d['Xtp'], Mtp=d['Mtp']),
        
        eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        
        eq.eqCPI(CPI = d['CPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        eq.eqRreal(Rreal=d['Rreal'],R=d['R'], CPI=d['CPI']), #expected GDPPI time series
        
        eq.eqsD(sD=d['sD'], Ij=d['Ij'], pCj=d['pCj'], Mj=d['Mj'], Xj=d['Xj'], pXj=d['pXj'], GDP=d['GDP']),
        
        #eq.eqT(T=d['T'], tauYj=d['tauYj'], pYj=d['pYj'], Yj=d['Yj'], tauSj=d['tauSj'], pCj=d['pCj'], Yij=d['Yij'], Cj=d['Cj'], Gj=d['Gj'], tauL=d['tauL'], pL=d['pL'], L=d['L']),
        
        eq.eqPriceTax(pGross=d['pCj'], pNet=d['pSj'], tau=d['tauSj']),

        eq.eqPriceTax(pGross=d['pL'], pNet=d['w'], tau=d['tauL']),

        ])

len_var=len(np.hstack([x.flatten() for x in list(variables.values())]))
len_sys=len(system(variables,parameters))

if len_var != len_sys:
    print("system has ",len_sys, " equations")
    print("there are ",len_var, " variables")
    sys.exit()



#put the bounds in the good format for the solver
def multiply_bounds_len(key,this_bounds,this_variables):
    return [this_bounds[key] for i in range(len(this_variables[key].flatten()))]

def bounds_dict(this_bounds,this_variables):
    return dict((k, multiply_bounds_len(k,this_bounds,this_variables) ) for k in this_variables.keys())

def flatten_bounds_dict(this_bounds,this_variables):
    return np.vstack(list(bounds_dict(this_bounds,this_variables).values()))

bounds_variables = [[row[i] for row in flatten_bounds_dict(bounds,variables)] for i in (0,1)]

sol= dict_least_squares(system, variables, parameters, bounds_variables, verb=2)

# bounds_variables = flatten_bounds_dict(bounds, variables)
# mybounds=MyBounds(bounds_variables)
# sol= dict_minimize(system, variables, parameters, mybounds.bounds)

#solFS= dict_fsolve(system, variables, parameters)

print("\n system of equations calculated at the solution found by least_square (an array of zeros is expected): \n \n", system(sol.dvar, parameters))
print(max(abs( system(sol.dvar, parameters))))

d = {**sol.dvar, **parameters}


#EXPORT TO CSV
augment_dict(sol.dvar, 'Gj', non_zero_index_G,'Gjn0')
augment_dict(sol.dvar, 'Ij', non_zero_index_I, 'Ijn0')
augment_dict(sol.dvar, 'Cj', non_zero_index_C, 'Cjn0')
augment_dict(sol.dvar, 'Xj', non_zero_index_X, 'Xjn0')
augment_dict(sol.dvar, 'Mj', non_zero_index_M, 'Mjn0')
augment_dict2D(sol.dvar, 'Yij', non_zero_index_Yij, 'Yijn0')
    
    
keysN=[k for k, v in sol.dvar.items() if np.shape(v) == np.shape(sol.dvar["Kj"])]
keys1=[k for k, v in sol.dvar.items() if np.shape(v) == np.shape(sol.dvar["R"])]


sol_N=pd.DataFrame({ key: sol.dvar[key] for key in keysN })
sol_1=pd.DataFrame({ key: sol.dvar[key] for key in keys1 })
sol_Yij=pd.DataFrame(sol.dvar["Yij"])

sol_N.to_csv("results/classic_N.csv")
sol_1.to_csv("results/classic_1.csv")
sol_Yij.to_csv("results/classic_Yij.csv")    


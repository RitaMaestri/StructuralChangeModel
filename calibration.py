import numpy as np
import sys
from data_calibration_from_matrix import variables, parameters, bounds, N
from import_csv import non_zero_index_G, non_zero_index_I, non_zero_index_C, non_zero_index_X, non_zero_index_M, non_zero_index_Yij

sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve, dict_basinhopping, MyBounds


#verify data 
cost=parameters['pL*Lj']+parameters['pK*Kj']+parameters['pSj*Yij'].sum(axis=0)+parameters['pMj*Mj']
revenue=parameters['pCj*Cj']+parameters['pCj*Gj']+parameters['pCj*Ij']+parameters['pXj*Xj']+parameters['pSj*Yij'].sum(axis=1)
if not all(cost-revenue)<2:
    print("Data not at equilibrium")
    sys.exit()
    
    
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

# def augment_dict(
#     d: 'dict[str, np.ndarray]', key0: str, keys: 'list[str]'
# ) -> None:  # Dictionary upload works inplace.
#     d[key0] = np.hstack([d[k] for k in keys])



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
    #augment_dict(d, 'wGj',non_zero_index_G ,'wGjn0')
    #augment_dict(d, 'wIj',non_zero_index_I, 'wIjn0')
    #augment_dict(d, 'alphaCj', non_zero_index_C, 'alphaCjn0')
    augment_dict(d, 'alphaXj', non_zero_index_X, 'alphaXjn0')
    augment_dict(d, 'betaMj', non_zero_index_M, 'betaMjn0')
    
    augment_dict2D(d, 'aYij', non_zero_index_Yij, 'aYijn0')
    augment_dict2D(d, 'Yij', non_zero_index_Yij, 'Yijn0')
    
    
    return np.hstack([
        
        eq.eqKLj(KLj =d['KLj'], bKL=d['bKL'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),#e-5
        
        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
        
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj'], _index=non_zero_index_Yij),
        
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']), #0.005%
        
        eq.eqpYj(pYj=d['pYj'],pSj=d['pSj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij']), #0.005%
        
        eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], _index=non_zero_index_X),#e-5
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj']),
       
        eq.eqCET(Zj=d['Yj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], Xj=d['Xj'], Yj=d['Dj'], sigmaj=d['sigmaXj']),#e-5
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj']),
        
        eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'],_index=non_zero_index_M),
        
        eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj']),
        
        #eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        #eq.eqwB(X=d['B'],wX=d['wB'],GDP=d['GDP']),
        
        #eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],R=d['R'], _index=non_zero_index_C),
        
        #eq.eqR(R=d['R'], Cj=d['Cj'], pCj=d['pCj']),
        
        #eq.eqw(pXj=d['pCj'],Xj=d['Gj'],wXj=d['wGj'],GDP=d['GDP'], _index=non_zero_index_G),
        
        #eq.eqw(pXj=d['pCj'],Xj=d['Ij'],wXj=d['wIj'],GDP=d['GDP'], _index=non_zero_index_I),
        
        eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
        
        #eq.eqF(F=d['L'],Fj=d['Lj']),
        
        #eq.eqF(F=d['K'],Fj=d['Kj']),
        
        eq.eqID(x=d['pCj'],y=d['pSj']), #e-10
        
        eq.eqID(x=d['pXj'],y=d['pMj']),
        
        #eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        eq.eqCalibi(pX=d['pXj'], Xj=d['Xj'], data=d['pXj*Xj'], _index=non_zero_index_X),#
        
        eq.eqCalibi(pX=d['pMj'], Xj=d['Mj'], data=d['pMj*Mj'],_index=non_zero_index_M),#3%
        
        eq.eqCalibi(pX=d['pCj'], Xj=d['Cj'], data=d['pCj*Cj'],_index=non_zero_index_C),#11%
        
        eq.eqCalibi(pX=d['pYj'], Xj=d['Yj'], data=d['pYj*Yj']),
        
        eq.eqCalibi(pX=d['pCj'], Xj=d['Gj'], data=d['pCj*Gj'], _index=non_zero_index_G),
        
        eq.eqCalibi(pX=d['pCj'], Xj=d['Ij'], data=d['pCj*Ij'], _index=non_zero_index_I),#3%
        
        eq.eqCalibi(pX=d['pL'], Xj=d['Lj'], data=d['pL*Lj']),#12
        
        eq.eqCalibi(pX=d['pK'], Xj=d['Kj'], data=d['pK*Kj']),#8%
        
        eq.eqCalibij(pYi=d['pSj'], Yij=d['Yij'], data=d['pSj*Yij'], _index=non_zero_index_Yij),#, _index=non_zero_index_Yij),#19%
        
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
sol= dict_least_squares(system, variables, parameters, bounds_variables)


# bounds_variables = flatten_bounds_dict(bounds, variables)
# sol= dict_minimize(system, variables, parameters, bounds_variables)


# bounds_variables = flatten_bounds_dict(bounds, variables)
# mybounds=MyBounds(bounds_variables)
# sol=dict_basinhopping(system, variables, parameters, mybounds)



print("\n system of equations calculated at the solution found by least_square (an array of zeros is expected): \n \n", system(sol.dvar, parameters))
print(sum(abs( system(sol.dvar, parameters))))



d = {**sol.dvar, **parameters}



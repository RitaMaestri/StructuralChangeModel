####### define closure : "johansen" , "neoclassic", "kaldorian", "keynes-marshall", "keynes", "keynes-kaldor","neokeynesian1", "neokeynesian2"   ########
closure="keynes"

import numpy as np
import pandas as pd
import sys
import simple_calibration as cal
from data_closures import bounds, N, variablesSystem
from import_csv import non_zero_index_G, non_zero_index_I, non_zero_index_C, non_zero_index_X, non_zero_index_M, non_zero_index_Yij
import import_csv as imp
sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve, dict_basinhopping, MyBounds, to_dict


system=variablesSystem(closure)

variables=system.endogeouns_dict

parameters=system.exogenous_dict

#####   DYNAMICS   ######

GDPgrowth=0.1

Lgrowth=0.1


####### check for errors ########

for k in (pavars := {**parameters, **variables}).keys():
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
    for var in variables[k].flatten():
        if var < bounds[k][0] or var > bounds[k][1]:
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

    common_equations= np.hstack([
        eq.eqKLj(KLj =d['KLj'],bKL=d['bKL'], bKLj=d['bKLj'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
                
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj'], _index=non_zero_index_Yij),
        
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']),
        
        eq.eqpYj(pYj=d['pYj'],pCj=d['pCj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], tauYj=d['tauYj']),
        
        eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], _index=non_zero_index_X),#e-5
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj']),
        
        eq.eqCESprice(pZj=d['pYj'],pXj=d['pXj'],pYj=d['pDj'],alphaXj=d['alphaXj'],alphaYj=d['alphaDj'],sigmaj=d['sigmaXj']),
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj']),
        
        eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'],_index=non_zero_index_M),
        
        eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj']),
        
        eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
                
        eq.eqID(x=d['pXj'],y=d['pMj']),
        
        eq.eqCobbDouglasj(Qj=d['Cj'],alphaQj=d['alphaCj'],pCj=d['pCj'],Q=d['R'], _index=non_zero_index_C),
        
        eq.eqCobbDouglasj(Qj=d['Gj'],alphaQj=d['alphaGj'],pCj=d['pCj'],Q=d['G'], _index=non_zero_index_G),
        
        eq.eqCobbDouglasj(Qj=d['Ij'],alphaQj=d['alphaIj'],pCj=d['pCj'],Q=d['I'], _index=non_zero_index_I),
        
        eq.eqw(X=d['G'],wX=d['wG'],GDP=d['GDP']),
        
        eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
    
        eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pXj=d['pXj'], pCtp= d['pCtp'], pXtp=d['pXtp'], Cj= d['Cj'], Gj= d['Gj'], Ij= d['Ij'], Xj=d['Xj'], Mj=d['Mj'], Ctp= d['Ctp'], Gtp= d['Gtp'], Itp= d['Itp'], Xtp=d['Xtp'], Mtp=d['Mtp']),
        
        eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        
        eq.eqCPI(CPI = d['CPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        eq.eqRreal(Rreal=d['Rreal'],R=d['R'], CPI=d['CPI']), #expected GDPPI time series
        
        eq.eqT(T=d['T'], tauYj=d['tauYj'], pYj=d['pYj'], Yj=d['Yj'], tauSj=d['tauSj'], pSj=d['pSj'], Sj=d['Sj'], tauL=d['tauL'], w=d['w'], Lj=d['Lj']),
        
        eq.eqPriceTax(pGross=d['pCj'], pNet=d['pSj'], tau=d['tauSj']),
        
        eq.eqPriceTax(pGross=d['pL'], pNet=d['w'], tau=d['tauL'])
        
        ])
    

    if closure=="johansen": 
        return np.hstack([common_equations,
                          np.hstack([
                                        eq.eqsD(sD=d['sD'], Ij=d['Ij'], pCj=d['pCj'], Mj=d['Mj'], Xj=d['Xj'], pXj=d['pXj'], GDP=d['GDP']),
                                        
                                        eq.eqw(X=d['I'],wX=d['wI'],GDP=d['GDP']),
                                        
                                        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                        
                                        eq.eqF(F=d['L'],Fj=d['Lj']),
                                        
                                        eq.eqF(F=d['K'],Fj=d['Kj']),
                                        
                                        eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),

                                        
                                        ])
                        ])

    elif closure=="neoclassic":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], K=d['K'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.eqF(F=d['L'],Fj=d['Lj']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),

                                                                            
                          ])
                         ])
    
    elif closure=="kaldorian":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqw(X=d['I'],wX=d['wI'],GDP=d['GDP']),
                                      
                                      eq.eqF(F=d['L'],Fj=d['Lj']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),

                                    ])
                         ])
    elif closure=="keynes-marshall":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.eqw(X=d['I'],wX=d['wI'],GDP=d['GDP']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),

                                     ])
                         ])
    
    elif closure=="keynes" or closure=="keynes-kaldor" :
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqw(X=d['I'],wX=d['wI'],GDP=d['GDP']),
                                      
                                      
                                      eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                                      
                                      eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                                      
                                      eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),

                                      ])
                         ])

    elif closure=="keynes-marshall":
        return np.hstack([common_equations,
                          np.hstack([
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.eqw(X=d['I'],wX=d['wI'],GDP=d['GDP']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),

                                     ])
                         ])
    
    elif closure=="neokeynesian1":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                                      
                                      eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                                      
                                      eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                                      
                                      eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                                      
                                      eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                                      
                                      eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),

                                      eq.eqw(X=d['B'],wX=d['wB'],GDP=d['GDP']),
                          
                          ])
                         ])    
    
    
    elif closure=="neokeynesian1":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                                      
                                      eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                                      
                                      eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                                      
                                      eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                                      
                                      eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                                      
                                      eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),

                                                                            
                          ])
                         ])   
    
    elif closure=="neokeynesian2":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqI(I=d['I'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], G=d['G'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                                      
                                      eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                                      
                                      eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                                      
                                      eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                                      
                                      eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                                      
                                      eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),

                                      eq.eqIneok(I=d['I'], K=d['K'], alphaIK=d['alphaIK'] )                                    
                          ])
                         ])   
    
    else:
        print("the closure doesn't exist")
        sys.exit()


#put the bounds in the good format for the solver
def multiply_bounds_len(key,this_bounds,this_variables):
    return [this_bounds[key] for i in range(len(this_variables[key].flatten()))]

def bounds_dict(this_bounds,this_variables):
    return dict((k, multiply_bounds_len(k,this_bounds,this_variables) ) for k in this_variables.keys())

def flatten_bounds_dict(this_bounds,this_variables):
    return np.vstack(list(bounds_dict(this_bounds,this_variables).values()))


#####  create a reduced dictionary for variables (without zeros)  #####

variables_values = [ variables[keys][variables[keys]!=0] for keys in variables.keys()]

var_keys=list(variables.keys())

non_zero_variables = {var_keys[i]: variables_values[i] for i in range(len(var_keys))}

bounds_variables = [[row[i] for row in flatten_bounds_dict(bounds, non_zero_variables)] for i in (0,1)]


#### Calibration check ####

if max(system(variables, parameters))>1e-07:
    d={**variables,**parameters}
    print("the system is not correctly calibrated")
    sys.exit()
    

##### Growth  #####

parameters["GDPreal"]= parameters["GDPreal"]*(1+GDPgrowth)

#parameters["L"]= parameters["L"]*(1+Lgrowth)


#####  disrupt the initial condition #####
variables["R"]=variables["R"]*0.9
variables["I"]=variables["I"]*0.9
variables["Kj"]=variables["Kj"]*0.9
variables["Yj"]=variables["Yj"]*0.9
variables["Cj"]=variables["Cj"]*0.9


######  SYSTEM SOLUTION  ######

sol= dict_least_squares(system, variables, parameters, bounds_variables, verb=2, check=False)





# bounds_variables = flatten_bounds_dict(bounds, variables)
# mybounds=MyBounds(bounds_variables)
# sol= dict_minimize(system, variables, parameters, mybounds.bounds)



print("\n \n", closure," closure \n max of the system of equations calculated at the solution: \n")
print(max(abs( system(sol.dvar, parameters))))

d = {**sol.dvar, **parameters}









#EXPORT TO CSV
# #to_dict(sol.dvar,variables,is_variable=True)

# augment_dict(sol.dvar, 'Gj', non_zero_index_G,'Gjn0')
# augment_dict(sol.dvar, 'Ij', non_zero_index_I, 'Ijn0')
# augment_dict(sol.dvar, 'Cj', non_zero_index_C, 'Cjn0')
# augment_dict(sol.dvar, 'Xj', non_zero_index_X, 'Xjn0')
# augment_dict(sol.dvar, 'Mj', non_zero_index_M, 'Mjn0')
# augment_dict2D(sol.dvar, 'Yij', non_zero_index_Yij, 'Yijn0')
    
    
# keysN=[k for k, v in sol.dvar.items() if np.shape(v) == np.shape(sol.dvar["Kj"])]
# keys1=[k for k, v in sol.dvar.items() if np.shape(v) == np.shape(sol.dvar["R"])]


# sol_N=pd.DataFrame({ key: sol.dvar[key] for key in keysN })
# sol_1=pd.DataFrame({ key: sol.dvar[key] for key in keys1 })
# sol_Yij=pd.DataFrame(sol.dvar["Yij"])

# sol_N.to_csv("results/classic_N.csv")
# sol_1.to_csv("results/classic_1.csv")
# sol_Yij.to_csv("results/classic_Yij.csv")    


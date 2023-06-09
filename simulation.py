import numpy as np
import pandas as pd
import sys
from data_closures import bounds, N, calibrationDict
from import_GTAP_data import non_zero_index_G, non_zero_index_I, non_zero_index_C, non_zero_index_X, non_zero_index_M, non_zero_index_Yij
sys.path.append('/home/rita/Documents/Stage/Code')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve, dict_basinhopping, MyBounds, to_dict
from time_series_data import sys_df, growth_ratio_to_rate
from datetime import datetime
import random
import math
import copy
import json
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")

###### PARAMETERS SETTING ###########

# closure : "johansen" , "neoclassic", "kaldorian", "keynes-marshall", "keynes", "keynes-kaldor","neokeynesian1", "neokeynesian2"   ########
closure="johansen"

#stop=2017
#step=1
#years = np.array(range(2015, stop+1, step))
add_string=""

growth_ratios_df = pd.read_csv("data/REMIND_exogenous_data.csv", index_col="variable")#[years.astype(str)]

growth_ratios=growth_ratios_df.T.reset_index().to_dict(orient='list')
growth_ratios.pop("index")
growth_ratios={k:np.array(v) for k,v in growth_ratios.items()}

years = np.array([eval(i) for i in growth_ratios_df.columns])
stop=years[-1]

Lg_rate = growth_ratio_to_rate( growth_ratios["L"])

dynamic_parameters={}
#dynamic_parameters_df= pd.read_csv("data/INSEE FRA/dynamic parameters.csv", index_col="variable")[years.astype(str)]

#dynamic_parameters={
#    "GDPreal":np.array(dynamic_parameters_df.loc["GDPreal"]),
#`    }

endoKnext = False if 'K' in {**growth_ratios,**dynamic_parameters}.keys() else True
endostring="endoKnext" if endoKnext else "exoKnext"

name = str().join(["results/",closure,str(2015),"-",str(stop),endostring,add_string,"(",dt_string,")",".csv"])

######### CALIBRATION AND MODEL SOLUTION #############

calibration = calibrationDict(closure, Lg_rate[0], endoKnext)

variables_calibration = calibration.endogeouns_dict

parameters_calibration = calibration.exogenous_dict

## Build the systen ##

System=sys_df(years, growth_ratios, variables_calibration, parameters_calibration)

####### check for errors ########

for item in variables_calibration.keys():
    if item in parameters_calibration:
        print("the same variable is both in variables and parameters")
        sys.exit()

for k in parameters_calibration.keys():
    for par in np.array([parameters_calibration[k]]).flatten() :
        if par < bounds[k][0] or par > bounds[k][1]:
            print("parameter ", k ," out of bounds")
            sys.exit()

for k in variables_calibration.keys():
    for var in  np.array([variables_calibration[k]]).flatten():
        if var < bounds[k][0] or var > bounds[k][1]:
            print("variable ", k ," out of bounds")
            sys.exit()

def system(var, par, write=True):

    d = {**var, **par}
    
    if write:
        np.savez('var.npz', **var)
        np.savez('par.npz', **par)
        
    common_equations= np.hstack([
        eq.eqKLj(KLj =d['KLj'],bKL=d['bKL'], bKLj=d['bKLj'], Lj=d['Lj'], Kj=d['Kj'], alphaLj=d['alphaLj'], alphaKj=d['alphaKj']),
                
        eq.eqFj(Fj=d['Kj'],pF=d['pK'],KLj=d['KLj'],pKLj=d['pKLj'],alphaFj=d['alphaKj']),        
        
        eq.eqYij(Yij=d['Yij'],aYij=d['aYij'],Yj=d['Yj'], _index=non_zero_index_Yij),
        
        eq.eqKL(KLj=d['KLj'],aKLj=d['aKLj'],Yj=d['Yj']),
        
        eq.eqpYj(pYj=d['pYj'],pCj=d['pCj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], tauYj=d['tauYj']),
        
        eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], _index=non_zero_index_X),#e-5
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj']),
        
        eq.eqCESprice(pZj=d['pYj'], pXj=d['pXj'], pYj=d['pDj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], sigmaj=d['sigmaXj']),
        
        eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj']),
        
        eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'],_index=non_zero_index_M),
        
        eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj']),
        
        eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
                
        eq.eqID(x=d['pXj'],y=d['pMj']),
        
        eq.eqCobbDouglasj(Qj=d['Cj'],alphaQj=d['alphaCj'],pCj=d['pCj'],Q=d['R'], _index=non_zero_index_C),
        
        eq.eqCobbDouglasj(Qj=d['Gj'],alphaQj=d['alphaGj'],pCj=d['pCj'],Q=d['Rg'], _index=non_zero_index_G),
        
        eq.eqIj(Ij=d['Ij'],alphaIj=d['alphaIj'],I=d['I'],_index=non_zero_index_I),
        
        eq.eqMult(result=d['Rg'],mult1=d['wG'],mult2=d['GDP']),
        
        eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
    
        eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pXj=d['pXj'], pCtp= d['pCtp'], pXtp=d['pXtp'], Cj= d['Cj'], Gj= d['Gj'], Ij= d['Ij'], Xj=d['Xj'], Mj=d['Mj'], Ctp= d['Ctp'], Gtp= d['Gtp'], Itp= d['Itp'], Xtp=d['Xtp'], Mtp=d['Mtp']),
        
        eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        
        eq.eqCPI(CPI = d['CPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        eq.eqRreal(Rreal=d['Rreal'],R=d['R'], CPI=d['CPI']), #expected GDPPI time series
        
        eq.eqT(T=d['T'], tauYj=d['tauYj'], pYj=d['pYj'], Yj=d['Yj'], tauSj=d['tauSj'], pSj=d['pSj'], Sj=d['Sj'], tauL=d['tauL'], w=d['w'], Lj=d['Lj']),#
        
        eq.eqPriceTax(pGross=d['pCj'], pNet=d['pSj'], tau=d['tauSj']),
        
        eq.eqPriceTax(pGross=d['pL'], pNet=d['w'], tau=d['tauL']),
        
        eq.eqpI(pI=d['pI'],pCj=d['pCj'],alphaIj=d['alphaIj']),
        
        eq.eqMult(result=d['Ri'],mult1=d['pI'],mult2=d['I']),
        
        ])
    
    common_equations = np.hstack([common_equations, eq.eqinventory(Knext=d['Knext'], K=d['K'], delta=d['delta'], I=d['I'])]) if endoKnext else common_equations
                      

    if closure=="johansen": 
        solution = np.hstack([common_equations,
                          np.hstack([
                                        eq.eqsD(sD=d['sD'], Ij=d['Ij'], pCj=d['pCj'], Mj=d['Mj'], Xj=d['Xj'], pXj=d['pXj'], GDP=d['GDP']),
                                        
                                        eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                                        
                                        eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                        
                                        eq.eqF(F=d['L'],Fj=d['Lj']),
                                        
                                        eq.eqF(F=d['K'],Fj=d['Kj']),
                                        
                                        eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                                    ])
                        ])
        
        #print("cost=",0.5 * sum(solution**2))
        
        return solution 

    elif closure=="neoclassic":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.eqF(F=d['L'],Fj=d['Lj']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),

                                                                            
                          ])
                         ])
    
    elif closure=="kaldorian":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                                      eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                                      
                                      eq.eqF(F=d['L'],Fj=d['Lj']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                        
                                    ])
                         ])
    elif closure=="keynes-marshall":
        return np.hstack([common_equations,
                          np.hstack([
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),

                                     ])
                         ])
    
    elif closure=="keynes" or closure=="keynes-kaldor" :
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                                      eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                                      
                                      eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                                      
                                      eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                                      
                                      eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),

                                      ])
                         ])

    elif closure=="keynes-marshall":
        return np.hstack([common_equations,
                          np.hstack([
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.eqMult(result=d['I'],mult1=d['wI'],mult2=d['GDP']),
                                      
                                      eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                                      eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),

                                     ])
                         ])
    
    elif closure=="neokeynesian1":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                                      eq.eqFj(Fj=d['Lj'],pF=d['pL'],KLj=d['KLj'],pKLj= d['pKLj'],alphaFj=d['alphaLj']),
                                      
                                      eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                                      
                                      eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                                      
                                      eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                                      
                                      eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                                      
                                      eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                                      
                                      eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),

                                      eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          
                          ])
                         ])   
     
    
    elif closure=="neokeynesian2":
        return np.hstack([common_equations,        
                          np.hstack([
                                      eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
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
        
#### Calibration check ####

if max(system(variables_calibration, parameters_calibration, False))>1e-07:
    d={**variables_calibration,**parameters_calibration}
    print("the system is not correctly calibrated")
    d = {**variables_calibration, **parameters_calibration}
    sys.exit()

#### set the bounds in the good format for the solver ####
def multiply_bounds_len(key,this_bounds,this_variables):
    return [this_bounds[key] for i in range(len(this_variables[key].flatten()))]

def bounds_dict(this_bounds,this_variables):
    return dict((k, multiply_bounds_len(k,this_bounds,this_variables) ) for k in this_variables.keys())

def flatten_bounds_dict(this_bounds,this_variables):
    return np.vstack(list(bounds_dict(this_bounds,this_variables).values()))


#####  create a reduced dictionary for variables (without zeros) and correspondent set of bounds#####

def to_array(candidate):
    return candidate if isinstance(candidate, np.ndarray) else np.array([candidate])

variables_values = [ to_array(variables_calibration[keys])[to_array(variables_calibration[keys]) !=0 ] for keys in variables_calibration.keys()]

var_keys=list(variables_calibration.keys())

non_zero_variables = {var_keys[i]: variables_values[i] for i in range(len(var_keys))}

bounds_variables = [[row[i] for row in flatten_bounds_dict(bounds, non_zero_variables)] for i in (0,1)]



def kick(variables, number_modified=10, percentage_modified=0.1, modification=0.1):
    kicked_variables = copy.deepcopy(variables)
    keys = random.sample(list(variables.keys()), k =number_modified)
    for v_key in keys:
        if hasattr(variables[v_key], "__len__"):
            v_len=len(variables[v_key])
            sec = random.sample(range(v_len), k = math.ceil(v_len*percentage_modified))
            new_values= [(1+random.choice((-1, 1))*modification)*kicked_variables[v_key][i] for i in sec]
            
            kicked_variables[v_key][sec]=new_values
#            print("v_key:", v_key, "\n",
#                  "sec:", sec, "\n",
#                  "kicked ", kicked_variables[v_key][sec], "\n",
#                  "variables ", variables[v_key][sec], "\n")
        else: 
            new_values= (1+random.choice((-1, 1))*modification)*kicked_variables[v_key]
            kicked_variables[v_key]= new_values
#            print("v_key:", v_key, "\n",
#                  "kicked ", kicked_variables[v_key], "\n",
#                  "variables ", variables[v_key], "\n")
        return kicked_variables


######  SYSTEM SOLUTION  ######

#max(abs( system( kick(variables_calibration, number_modified=2, percentage_modified=0.1, modification=0.01), parameters_calibration)))
print("I got here")
sol = dict_least_squares( system, variables_calibration, parameters_calibration, bounds_variables, N, verb=2, check=True )

#maxerror=max(abs( system(variables_calibration, parameters_calibration)))
sys.exit()

for t in range(len(years[:-1])):
    variables=System.df_to_dict(var=True, t=years[t])
    
    parameters=System.df_to_dict(var=False, t=years[t+1])

    sol = dict_least_squares( system, variables, parameters, bounds_variables, N, verb=0, check=False )
    
    maxerror=max(abs( system(sol.dvar, parameters)))
    
    if maxerror>1e-06:
        print("the system doesn't converge, maxerror=",maxerror)
        sys.exit()
    print("year: ", t)
    
    #print("\n \n", closure," closure \n max of the system of equations calculated at the solution: \n")
    #print(maxerror, "\n")
    
    System.dict_to_df(sol.dvar, years[t+1])
    
    if endoKnext and years[t]<years[-2] :
        System.evolve_K(t+1)

results=pd.concat([System.variables_df,System.parameters_df], ignore_index=False)

results.to_csv(name)
   
            
# bounds_variables = flatten_bounds_dict(bounds, variables)
# mybounds=MyBounds(bounds_variables)
# sol= dict_minimize(system, variables, parameters, mybounds.bounds)

# print("\n \n", closure," closure \n max of the system of equations calculated at the solution: \n")
# print(max(abs( system(sol.dvar, parameters))))

# d = {**sol.dvar, **parameters}


#EXPORT TO CSV

# def augment_dict(d, key0, indexes, value_key):
#     array=np.zeros(N)
#     array[indexes] = d[value_key]
#     d[key0] = array
    
# def augment_dict2D(d, key0, indexes, value_key):
#     array=np.zeros([N,N])
#     for i,j,k in zip(indexes[0], indexes[1], range(len(indexes[0]))):
#             array[i,j]=d[value_key][k]
#     d[key0] = array

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



array_zero_keys = []
scalar_zero_keys = []

for key, value in variables.items():
    if isinstance(value, list) and 0 in value:
        array_zero_keys.append(key)
    elif isinstance(value, (int, float)) and value == 0:
        scalar_zero_keys.append(key)

print("Keys with at least one zero array element:", array_zero_keys)
print("Keys with zero scalar value:", scalar_zero_keys)


loaded_data = np.load('var.npz')
loaded_dict_var = {key: loaded_data[key] for key in loaded_data.files}

# Close the loaded_data object to release the resources
loaded_data.close()

loaded_data = np.load('par.npz')
loaded_dict_par = {key: loaded_data[key] for key in loaded_data.files}

# Close the loaded_data object to release the resources
loaded_data.close()

import numpy as np
import pandas as pd
import sys
from data_closures import bounds, N, calibrationDict
from import_GTAP_data import non_zero_index_G, non_zero_index_I, non_zero_index_C, non_zero_index_X, non_zero_index_M, non_zero_index_Yij,non_zero_index_L,non_zero_index_K
sys.path.append('/home/rita/Documents/Stage/Code/REMIND energy coupling')
import model_equations as eq
from solvers import dict_minimize, dict_least_squares, dict_fsolve, dict_basinhopping, MyBounds, to_dict
from time_series_data import sys_df, growth_ratio_to_rate
from datetime import datetime
import random
import math
import copy
#import json
import collections
from simple_calibration import A,M,SE,E,ST,CH,T
import warnings
warnings.filterwarnings("ignore")


now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
exogenous_data="REMIND_exogenous_data"
###### PARAMETERS SETTING ###########

# closure : "johansen" , "neoclassic", "kaldorian", "keynes-marshall", "keynes", "keynes-kaldor","neokeynesian1", "neokeynesian2"   ########
closure="johansen"

#stop=2017
#step=1
#years = np.array(range(2015, stop+1, step))
add_string="REMIND-"+ str(N) +"sectors"

growth_ratios_df = pd.read_csv("data/"+exogenous_data+".csv", index_col="variable")#[years.astype(str)]
growth_ratios_df_T = growth_ratios_df.T.reset_index().drop(columns="index")

counter=collections.Counter(list(growth_ratios_df.index))

scalar_growth_ratios = [k for k,v in counter.items() if v == 1]

growth_ratios=growth_ratios_df_T[growth_ratios_df_T.columns & scalar_growth_ratios].to_dict(orient='list')
growth_ratios={k:np.array(v) for k,v in growth_ratios.items()}

vector_growth_ratios =[k for k,v in counter.items() if v > 1]

for key in vector_growth_ratios:
    gr_array=growth_ratios_df_T[key].to_numpy().T
    growth_ratios[key]=gr_array

years = np.array([eval(i) for i in growth_ratios_df.columns])
stop=years[-1]
start=years[0]
Lg_rate = growth_ratio_to_rate( growth_ratios["L"])


dynamic_parameters={}
#dynamic_parameters_df= pd.read_csv("data/INSEE FRA/dynamic parameters.csv", index_col="variable")[years.astype(str)]

#dynamic_parameters={
#    "GDPreal":np.array(dynamic_parameters_df.loc["GDPreal"]),
#`    }

endoKnext = False if 'K' in {**growth_ratios,**dynamic_parameters}.keys() else True
endostring="endoKnext" if endoKnext else "exoKnext"

name = str().join(["results/",closure,str(start),"-",str(stop),endostring,add_string,"(",dt_string,")",".csv"])

######### CALIBRATION AND MODEL SOLUTION #############

calibration = calibrationDict(closure, Lg_rate[0], endoKnext)

variables_calibration = calibration.endogeouns_dict

parameters_calibration = calibration.exogenous_dict
## Build the systen ##

System=sys_df(years, growth_ratios, variables_calibration, parameters_calibration)




####### check for errors ########



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

def fill_nans(par_value, var_value):
    if isinstance(par_value, np.ndarray):
        par_copy = par_value.copy()
        if par_copy.ndim == 2:
            mask_par = np.isnan(par_copy)
            row_idx, col_idx = np.where(mask_par)
            par_copy[row_idx, col_idx] = var_value.flatten()
        else:
            mask_par = np.isnan(par_copy)
            par_copy[mask_par] = var_value
        return par_copy
    else:
        return var_value


def joint_dict(par, var):
    # Create a new dictionary to store the updated values
    d = par.copy()
    # Iterate through keys of A that are also present in B
    for key in var.keys() & par.keys():
        if np.isscalar(var[key]):
            d[key] = fill_nans(par[key], np.array([var[key]]))
        else:
            d[key] = fill_nans(par[key], var[key])
    return d


def system(var, par):

    d=joint_dict(par,var)
    global equations
    equations= {
        
        "eqCESquantityKj":eq.eqCESquantity(Xj=d['Kj'], Zj=d['KLj'], alphaXj=d['alphaKj'], alphaYj=d['alphaLj'], pXj=d['pK'], pYj=d['pL'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
        
        "eqCESpriceKL":eq.eqCESprice(pZj=d['pKLj'], pXj=d['pL'], pYj=d['pK'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta = d['bKL']),
        
        "eqYij":eq.eqYij_E(Yij=d['Yij'], aYij=d['aYij'],Yj=d['Yj'], lambda_KLM=d['lambda_KLM'], _index=non_zero_index_Yij),
        
        "eqKL":eq.eqKL(KLj=d['KLj'],aKLj=d['lambda_KLM']*d['aKLj'],Yj=d['Yj']),
        
        #"eqpYj":eq.eqpYj(pYj=d['pYj'],pCj=d['pCj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], tauYj=d['tauYj']),
        
        "eqpYj_E":eq.eqpYj_E(pYj=d['pYj'],pCj=d['pCj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], pY_Ej=d["pY_Ej"], tauYj=d['tauYj'], lambda_KLM=d['lambda_KLM']),
        
        "eqCESquantityX":eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'] , alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d["lambda_KLM"]*d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], thetaj=d['thetaj'],_index=non_zero_index_X),#e-5
        
        "eqCESquantityDy":eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj'],  thetaj=d['thetaj']),
        
        "eqCESpriceY":eq.eqCESprice(pZj=d['pYj'], pXj=d["lambda_KLM"]*d['pXj'], pYj=d['pDj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], sigmaj=d['sigmaXj'],  thetaj=d['thetaj']),
        
        "eqCESquantityDs":eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj'], thetaj=d['csij']),
        
        "eqCESquantityM":eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'], thetaj=d['csij'],_index=non_zero_index_M),
        
        "eqCESpriceS":eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj'], thetaj=d['csij']),
        
        "eqB":eq.eqB(B=d['B'],pXj=d["lambda_KLM"]*d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        "eqIDpX":eq.eqID(x=d['pXj'],y=d['pMj']),
        
        #"eqCobbDouglasjC":eq.eqCobbDouglasj(Qj=d['Cj'],alphaQj=d['alphaCj'],pCj=d['pCj'],Q=d['R'], _index=non_zero_index_C),
        
        "eqCobbDouglasjG":eq.eqCobbDouglasj(Qj=d['Gj'],alphaQj=d['alphaGj'],pCj=d['pCj'],Q=d['Rg'], _index=non_zero_index_G),
        
        "eqIj":eq.eqIj(Ij=d['Ij'],alphaIj=d['alphaIj'],I=d['I'],_index=non_zero_index_I),
        
        "eqMultRg":eq.eqMult(result=d['Rg'],mult1=d['wG'],mult2=d['GDP']),
        
        "eqSj":eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
        
        "eqGDP":eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d["lambda_KLM"]*d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        
        "eqGDPPI":eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pXj=d["lambda_KLM"]*d['pXj'], pCtp= d['pCtp'], pXtp=d['pXtp'], Cj= d['Cj'], Gj= d['Gj'], Ij= d['Ij'], Xj=d['Xj'], Mj=d['Mj'], Ctp= d['Ctp'], Gtp= d['Gtp'], Itp= d['Itp'], Xtp=d['Xtp'], Mtp=d['Mtp']),
        
        "eqGDPreal":eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        
        "eqCPI":eq.eqCPI(CPI = d['CPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        "eqRreal":eq.eqRreal(Rreal=d['Rreal'],R=d['R'], CPI=d['CPI']), #expected GDPPI time series
        
        "eqT":eq.eqT(T=d['T'], tauYj=d['tauYj'], pYj=d['pYj'], Yj=d['Yj'], tauSj=d['tauSj'], pSj=d['pSj'], Sj=d['Sj'], tauL=d['tauL'], w=d['w'], Lj=d['Lj']),#
        
        "eqPriceTaxtauS":eq.eqPriceTax(pGross=d['pCj'], pNet=d['pSj'], tau=d['tauSj'], exclude_idx=E),
        
        "eqPriceTaxtauL":eq.eqPriceTax(pGross=d['pL'], pNet=d['w'], tau=d['tauL']),
        
        "eqpI":eq.eqpI(pI=d['pI'],pCj=d['pCj'],alphaIj=d['alphaIj']),
        
        "eqMultRi":eq.eqMult(result=d['Ri'],mult1=d['pI'],mult2=d['I']),
        
        #energy coupling
        
        "eqC_E":eq.eqsum_scalar(d['Cj'][E], d['C_EB'],d['C_ET']),
        
        "eqY_E":eq.eqsum_arr(d['Yij'][E,:], d['YE_Pj'], d['YE_Bj'],d['YE_Tj'], d['YE_Ej']  ),
        
        "eqpC_E":eq.eqsum_pEYE(p_CE=d["pCj"][E], pY_Ej=d['pY_Ej'], C_E=d['Cj'][E], Y_Ej=d['Yij'][E,:], 
                               pE_B=d['pE_B'], C_EB=d['C_EB'], YE_Bj=d['YE_Bj'], pE_Pj=d['pE_Pj'], 
                               YE_Pj=d['YE_Pj'], pE_TnT=d['pE_TnT'], pE_TT=d['pE_TT'], C_ET=d['C_ET'], 
                               YE_Tj=d['YE_Tj'], pE_Ej=d['pE_Ej'], YE_Ej=d['YE_Ej']),
        
        # "eqE_P":eq.eqsum_scalar(d['E_P'], d['YE_Pj']),
        
        # "eqE_B":eq.eqsum_scalar(d['E_B'], d['YE_Bj'], d['C_EB']),
        
        # "eqE_T":eq.eqsum_scalar(d['E_T'], d['YE_Tj'], d['C_ET']),
        
        "eqCj_new":eq.eqCobbDouglasj_lambda(Qj=d['Cj'], alphaQj=d['alphaCj0'],pCj=d['pCj'],Q=d['R'], lambda_E=d["lambda_E"], lambda_nE=d["lambda_nE"], _index=non_zero_index_C),
        
        "eqlambda_nE":eq.eqlambda_nE(alphaCj=d['alphaCj0'],lambda_E=d['lambda_E'], lambda_nE=d['lambda_nE']),
        
        "eqpCE":eq.eqsum_pESE(p_SE=d['pSj'][E], tauSE=d['tauSj'][E], S_E=d['Sj'][E], Y_Ej=d['Yij'][E,:], C_E=d['Cj'][E], pY_Ej=d['pY_Ej'], p_CE=d['pCj'][E])
        }

    if endoKnext:
        equations.update("eqinventory", eq.eqinventory(Knext=d['Knext'], K=d['K'], delta=d['delta'], I=d['I']) )
                      

    if closure=="johansen": 
        equations.update({"eqsD":eq.eqsD(sD=d['sD'], Ij=d['Ij'], pCj=d['pCj'], Mj=d['Mj'], Xj=d['Xj'], pXj=d["lambda_KLM"]*d['pXj'], GDP=d['GDP']),
        
                                "eqMultwI":eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                                
                                "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
                                
                                "eqFL":eq.eqF(F=d['L'],Fj=d['Lj']),
                                
                                "eqFK":eq.eqF(F=d['K'],Fj=d['Kj']),
                                
                                "eqMultB":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP'])}
                                )
        
        solution = np.hstack(list(equations.values()))
        
        
        return solution 

    elif closure=="neoclassic":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
                                      
                          "eqL":eq.eqF(F=d['L'],Fj=d['Lj']),
                                      
                          "eqF":eq.eqF(F=d['K'],Fj=d['Kj']),
                                      
                          "eqMult":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP'])
                          })
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqIj"])
        return solution 

    elif closure=="kaldorian":
        equations.update({"eqlj":eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                           "eqRi": eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                            
                           "eqwI": eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                            
                           "eqL": eq.eqF(F=d['L'],Fj=d['Lj']),
                            
                           "eqK": eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                           "eqB": eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP'])})

        
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqIj"])
        return solution
    
    elif closure=="keynes-marshall":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
                            
                          "eqwI":eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                            
                          "eqK":eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                          "eqB":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP'])})
       
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqCESquantityX"])
        return solution


    
    elif closure=="keynes" or closure=="keynes-kaldor" :
        equations.update({"eqlj":eq.eqlj(l=d['l'], alphalj=d['alphalj'], KLj=d['KLj'], Lj=d['Lj']),
                                      
                          "eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                        
                          "eqwI":eq.eqMult(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
                            
                          "equL":eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                            
                          "eqw_real":eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                            
                          "eqw_curve":eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                            
                          "eqK":eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                          "eqB":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          })
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqIj"])
        return solution
    elif closure=="keynes-marshall":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                        
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL'], _index=non_zero_index_L),#e-5
                            
                          "eqwI":eq.eqMult(result=d['I'],mult1=d['wI'],mult2=d['GDP']),
                            
                          "eqK":eq.eqF(F=d['K'],Fj=d['Kj']),
                            
                          "eqB":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          })
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqIj"])
        return solution
    elif closure=="neokeynesian1":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL'], _index=non_zero_index_L),#e-5
                            
                          "equL":eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                            
                          "eqw_real":eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                            
                          "eqw_curve":eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                         
                          "equK":eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                            
                          "eqpK_real":eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                            
                          "eqsigmapK":eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),
        
                          "eqwB":eq.eqMult(result=d['B'],mult1=d['wB'],mult2=d['GDP']),
                          })                          
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqIj"])
        return solution
    elif closure=="neokeynesian2":
        equations.update({"eqRi":eq.eqRi(Ri=d['Ri'], sL=d['sL'], w=d['w'], Lj=d['Lj'], sK=d['sK'], Kj=d['Kj'], pK=d['pK'], sG=d['sG'], T=d['T'], Rg=d['Rg'], B=d['B']),
                                      
                          "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL'], _index=non_zero_index_L),#e-5
                            
                          "equL":eq.equ(u=d['uL'], L=d['L'], Lj=d['Lj']),
                            
                          "eqw_real":eq.eqw_real(w_real=d['w_real'], CPI=d['CPI'], w=d['w']),
                            
                          "eqw_curve":eq.eqw_curve(w_real=d['w_real'], alphaw=d['alphaw'], u=d['uL'], sigmaw=d['sigmaw'] ),
                            
                          "equK":eq.equ(u=d['uK'], L=d['K'], Lj=d['Kj']),
                            
                          "eqpK_real":eq.eqw_real(w_real=d['pK_real'], CPI=d['CPI'], w=d['pK']),
                            
                          "eqsigmapK":eq.eqw_curve(w_real=d['pK_real'], alphaw=d['alphapK'], u=d['uK'], sigmaw=d['sigmapK'] ),
    
                          "eqIneok":eq.eqIneok(I=d['I'], K=d['K'], alphaIK=d['alphaIK'] )
                          })
        solution = np.hstack(list(equations.values()))
        #print("equations Ij ", equations["eqIj"])
        return solution
    else:
        print("the closure doesn't exist")
        sys.exit()
        
#### Calibration check ####
max_err_cal=max(abs(system(variables_calibration, parameters_calibration)))
# print("maxerrcal", max_err_cal)

if max_err_cal>1e-07:
    d=joint_dict(parameters_calibration,variables_calibration)
    print("the system is not correctly calibrated")
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

var_keys = list(variables_calibration.keys())

non_zero_variables = {var_keys[i]: variables_values[i] for i in range(len(var_keys))}

bounds_variables = [[row[i] for row in flatten_bounds_dict(bounds, non_zero_variables)] for i in (0,1)]

#number_modified=10, percentage_modified=0.1, modification=0.1, seed=4

seed = random.randrange(sys.maxsize)

def kick(variables, number_modified=10, percentage_modified=0.1, modification=0.1, seed=seed):
    random.seed(seed)
    kicked_variables = copy.deepcopy(variables)
    keys = random.sample(list(variables.keys()), k=number_modified)
#    print("Variables kicked at:", keys,
#          "\n number modified: ", number_modified,
#          "\n percentage: ",percentage_modified,
#          "\n modification: ",modification,
#          "\n seed: ",seed,)

    for v_key in keys:
        if v_key in kicked_variables:
            if hasattr(kicked_variables[v_key], "__len__"):
                v_len = len(kicked_variables[v_key])
                sec = random.sample(range(v_len), k=math.ceil(v_len * percentage_modified))
                new_values= [(1+random.choice((-1, 1))*modification)*float(kicked_variables[v_key][i]) for i in sec]
                kicked_variables[v_key][sec] = new_values
            else:
                if kicked_variables[v_key] != 0:
                    new_value = (1 + random.choice((-1, 1)) * modification) * kicked_variables[v_key]
                    kicked_variables[v_key] = new_value

    return kicked_variables

######  SYSTEM SOLUTION  ######

maxerror=max(abs( system(variables_calibration, parameters_calibration)))
print(maxerror)
if maxerror>1e-5:
    print("the system is not at equilibrium")

#end = np.where(years == 2050)[0][0]
for t in range(len(years)):
    print("year: ", years[t])
    if t==0:
        variables=variables_calibration #kick(variables_calibration)

    else:
        variables=System.df_to_dict(var=True, t=years[t-1]) #kick(System.df_to_dict(var=True, t=years[t-1]))
    
    parameters=System.df_to_dict(var=False, t=years[t])
    
    sol = dict_least_squares( system, variables, parameters, bounds_variables, N, verb=1, check=True)
    
    maxerror=max(abs( system(sol.dvar, parameters)))
    
    system(sol.dvar, parameters)
    
    #System.dict_to_df(sol.dvar, years[t])

    if maxerror>1e-06:
        print("the system doesn't converge, maxerror=",maxerror)
        sys.exit()

    
    #print("\n \n", closure," closure \n max of the system of equations calculated at the solution: \n")
    #print(maxerror, "\n")

    System.dict_to_df(sol.dvar, years[t])
    
    if endoKnext and years[t]<years[-2] :
        System.evolve_K(t+1)
    if t>0 and t<len(years)-2:
        System.evolve_tp(t)

par_csv=copy.deepcopy(System.parameters_df)
var_csv=copy.deepcopy(System.variables_df)
# Identify the NaN values in DataFrame A
common_indexes = set(par_csv.index) & set(var_csv.index)

for i in common_indexes :
    index_positions = np.array([j for j, label in enumerate(par_csv.index) if label == i])
    if len(index_positions)>1:
        par_mask=par_csv.loc[i].isna().any(axis=1)
        par_csv.iloc[index_positions[par_mask]]=var_csv.loc[i]
    else:
        par_csv.iloc[index_positions]=var_csv.loc[i]

        
# Update the NaN values in DataFrame A with values from DataFrame B



#results=pd.concat([System.variables_df,System.parameters_df], ignore_index=False)
par_csv.to_csv(name)
   
            
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



# array_zero_keys = []
# scalar_zero_keys = []

# for key, value in variables.items():
#     if isinstance(value, list) and 0 in value:
#         array_zero_keys.append(key)
#     elif isinstance(value, (int, float)) and value == 0:
#         scalar_zero_keys.append(key)

# print("Keys with at least one zero array element:", array_zero_keys)
# print("Keys with zero scalar value:", scalar_zero_keys)


# loaded_data = np.load('var.npz')
# loaded_dict_var = {key: loaded_data[key] for key in loaded_data.files}

# # Close the loaded_data object to release the resources
# loaded_data.close()

# loaded_data = np.load('par.npz')
# loaded_dict_par = {key: loaded_data[key] for key in loaded_data.files}

# # Close the loaded_data object to release the resources
# loaded_data.close()

# f0 = loadtxt('f0.csv', delimiter=',')
# pars = loadtxt('par.csv', delimiter=',')

def compare_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        print("The dictionaries have different keys")
    else:
        equal_keys = True
        for key in dict1.keys():
            value1 = dict1[key]
            value2 = dict2[key]
            if type(value1) != type(value2):
                equal_keys = False
                print(f"The key '{key}' has different value types")
            elif hasattr(value1, "__len__"):
                if not np.array_equal(value1, value2):
                    equal_keys = False
                    if len(value1) != len(value2):
                        print(f"The key '{key}' has arrays with different lengths")
                    else:
                        unequal_indexes = np.where(value1 != value2)[0]
                        print(f"The key '{key}' has unequal values at indexes: {unequal_indexes}")
                        print(f"Value 1 at unequal indexes: {value1[unequal_indexes]}")
                        print(f"Value 2 at unequal indexes: {value2[unequal_indexes]}")
            else:
                if value1 != value2:
                    equal_keys = False
                    print(f"The key '{key}' has unequal float values")
        if equal_keys:
            print("The dictionaries are equal")


def count_elements(dictionary):
    count = 0
    for value in dictionary.values():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                count += len(value)
            elif value.ndim == 2:
                count += value.size
        elif isinstance(value, (float, int)):
            count += 1
        else:
            print("Unsupported type:", type(value))
    return count



def filter_nan_values(original_dict):
    new_dict = {}
    
    for key, value in original_dict.items():
        if isinstance(value, np.ndarray):
            if not np.any(np.isnan(value)):
                new_dict[key] = value
        elif np.isnan(value):
            continue
        else:
            new_dict[key] = value
    
    return new_dict

def compare_dictionaries(dict1, dict2):
    unequal_keys = []

    for key in dict1.keys():
        if key in dict2.keys():
            value1 = dict1[key]
            value2 = dict2[key]

            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                if not np.array_equal(value1, value2):
                    unequal_keys.append(key)
            elif value1 != value2:
                unequal_keys.append(key)
        else:
            unequal_keys.append(key)

    # Check for keys that exist in dict2 but not in dict1
    for key in dict2.keys():
        if key not in dict1.keys():
            unequal_keys.append(key)

    return unequal_keys


def compare_dictionaries_keys(dictionary1, dictionary2):
    # Find keys present only in dictionary1
    keys_only_in_dictionary1 = set(dictionary1.keys()) - set(dictionary2.keys())

    # Find keys present only in dictionary2
    keys_only_in_dictionary2 = set(dictionary2.keys()) - set(dictionary1.keys())

    return keys_only_in_dictionary1, keys_only_in_dictionary2


def find_keys_with_large_elements(dictionary):
    keys_with_large_elements = []

    for key, value in dictionary.items():
        # Check if at least one element in the array is greater than one
        if isinstance(value, (list, np.ndarray)):
            if np.any(abs(np.array(value)) > 4*10e-5):
                keys_with_large_elements.append(key)

    return keys_with_large_elements

def column(matrix, i):
    return [row[i] for row in matrix]
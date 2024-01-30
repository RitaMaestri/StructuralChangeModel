#equations cge
import numpy as np
from math import sqrt
from simple_calibration import A,M,SE,E,ST,CH,T
#import data_calibration_from_matrix as dt

#EQUATIONS

def eqKLj(KLj,bKL, bKLj, Lj, Kj, alphaLj, alphaKj):
    #print("eqKLj")

    zero = -1 + KLj / (
        bKL *bKLj*np.float_power(Lj,alphaLj) * np.float_power(Kj,alphaKj)
    )

    return zero


# same equation for Lj and Kj (Factors)

def eqFj(Fj,pF,KLj,pKLj,alphaFj):

    zero= -1 + Fj / (
        (np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0))/pF
    )

    return zero

def eqlj(l, alphalj, KLj, Lj):
    zero = - 1 + KLj * alphalj * l /Lj
    return zero

def eqYij(Yij,aYij,Yj, _index=None):

    Yjd=np.diag(Yj)
    
    if isinstance(_index, np.ndarray):
        #print("Yij check: ",(Yij[_index[0],_index[1]]==dt.variables['Yijn0']).all())
        
        zero= -1 + Yij[_index[0],_index[1]] / np.dot(aYij,Yjd)[_index[0],_index[1]]
    else:
        zero= -1 + Yij / np.dot(aYij,Yjd)
        zero=zero.flatten()
    #convert matrix to vector

    return zero


def eqKL(KLj,aKLj,Yj):

    zero=-1 + KLj / np.multiply(aKLj,Yj)

    return zero

# def eqKL_lambda(KLj,aKLj,Yj,lambda_KL):
#     comp_KLj = np.multiply(aKLj,Yj)
#     comp_KLj[E]=aKLj[E]*Yj[E]*lambda_KL
    
#     zero=-1 + KLj / comp_KLj

#     return zero

def eqpYj(pYj,pCj,aKLj,pKLj,aYij, tauYj):

    pCjd=np.diag(pCj)

    zero= -1 + pYj / (
        ( aKLj * pKLj + np.dot(pCjd,aYij).sum(axis=0) )*(1+tauYj) #AXIS=0 sum over the rows CHECKED
    )

    return zero


def eqpYj_E(pYj, pCj, aKLj, pKLj, aYij, pY_Ej, tauYj, lambda_KLM):
    
    pCjnE = np.delete(pCj, E)
    
    aYijnE = np.delete(aYij, (E), axis=0)
    
    pCjd=np.diag(pCjnE)
    
    zero= -1 + pYj / (
        ( lambda_KLM * aKLj * pKLj + lambda_KLM[E]*np.dot(pCjd,aYijnE).sum(axis=0) + aYij[E]*pY_Ej)*(1+tauYj) #AXIS=0 sum over the rows CHECKED
    )
    
    return zero

def eqCES(Zj, thetaj, alphaXj,alphaYj,Xj,Yj,sigmaj):

    etaj = (sigmaj-1)/sigmaj

    partj = alphaXj * np.float_power(Xj,etaj, out=np.zeros(len(Xj)),where=(Xj!=0)) + alphaYj * np.float_power(Yj,etaj)

    zero = -1 + Zj / ( np.float_power(partj, 1/etaj) * thetaj )

    return zero

#eqCET(Zj=d["Sj"], alphaXj=d["betaDj"],alphaYj=d["betaMj"],Xj=d["Dj"],Yj=d["Mj"],sigmaj=d["sigmaSj"])

def eqCESquantity(Xj, Zj, thetaj, alphaXj, alphaYj, pXj, pYj, sigmaj, _index=None, theta=1):
        
    if isinstance(_index, np.ndarray):
        Xj=Xj[_index]
        Zj=Zj[_index]
        alphaXj=alphaXj[_index]
        alphaYj=alphaYj[_index]
        pXj=pXj[_index]
        pYj=pYj[_index]
        sigmaj=sigmaj[_index]

    #is it correct??? TODO
    partj = np.float_power(alphaXj, sigmaj, out=np.zeros(len(alphaXj)),where=(alphaXj!=0)) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj, out=np.zeros(len(alphaYj)),where=(alphaYj!=0))* np.float_power(pYj,1-sigmaj)
    
    zero= -1 + Xj / (
        np.float_power(alphaXj/pXj, sigmaj) * np.float_power(partj , sigmaj/(1-sigmaj) ) * Zj * np.float_power(thetaj*theta,-1)
    )
    return zero


def eqCESprice(pZj,pXj,pYj,alphaXj,alphaYj,sigmaj, thetaj, theta=1):

    partj= np.float_power(alphaXj,sigmaj, out=np.zeros(len(alphaXj)),where=(alphaXj!=0)) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj, out=np.zeros(len(alphaYj)),where=(alphaYj!=0)) * np.float_power(pYj,1-sigmaj)

    zero= -1 + pZj / (  
        np.float_power(theta*thetaj, -1) *
        np.float_power(partj, 1/(1-sigmaj) )
    )

    return zero



def eqB(B,pXj,Xj,pMj,Mj):
    
    zero = -1 + B / sum(pXj*Xj-pMj*Mj)
    
    return zero


def eqMult(result, mult1, mult2):
    
    zero = -1 + result / (mult1*mult2)
    
    return(zero)



##check for the case where the index is an empty array! TODO
def eqCobbDouglasj(Qj,alphaQj,pCj,Q,_index=None):
    
    if isinstance(_index, np.ndarray):
        zero= -1 + Qj[_index] / ( alphaQj[_index] * (Q/ pCj[_index]) )
    else:
        zero= -1 + Qj / ( alphaQj * (Q/ pCj) )
    
    return zero



def eqalphaCj(alphaCj,R,pCj,alphaCDESj,betaRj, _index=None):
    if isinstance(_index, np.ndarray):
        zero= -1 + alphaCj[_index] / (alphaCDESj * np.float_power( R/pCj , betaRj ) / sum(alphaCDESj * np.float_power( R/pCj , betaRj )))[_index]
    else:
        zero= -1 + alphaCj / (alphaCDESj * np.float_power( R/pCj , betaRj ) / sum(alphaCDESj * np.float_power( R/pCj , betaRj )))

    return zero


def eqR(R,Cj,pCj):

    zero = -1 + R / sum(Cj*pCj)

    return zero / sum(Cj*pCj)

    return zero

#eq omegaG/I: fixed GDP fraction

# def eqw(pXj,Xj,wXj,GDP, _index = None):
    
#     if isinstance(_index, np.ndarray):
#         deno = (pXj * Xj)[_index]
#         nume = (wXj * GDP)[_index]
#     else:
#         deno = (pXj * Xj)
#         nume = (wXj * GDP)
    
#     zero = -1 + nume / deno

#     return zero


def eqTotalConsumptions(pCj, Qj, Q):
    
    zero= - 1 + sum(pCj*Qj)/ Q
    
    return zero



def eqSj(Sj,Cj,Gj,Ij,Yij):
    #print("eqSj")
    zero = -1 + Sj / (
        (Cj + Gj + Ij + Yij.sum(axis=1))#sum over the rows
    )

    return zero


#same equation for L and K

def eqF(F,Fj):
    #print("eqF")

  
    zero= -1 + F / sum(Fj)

    return zero


def eqID(x,y):
    #print("eqID")
    zero=-1 + x / y

    return zero


def eqGDP(GDP,pCj,Cj,Gj,Ij,pXj,Xj,pMj,Mj):
    #print("eqGDP")

    zero= -1 + GDP / sum(pCj*(Cj+Gj+Ij)+pXj*Xj-pMj*Mj)

    return zero

def eqGDP_E(GDP,pCj,Cj,Gj,Ij,pXj,Xj,pMj,Mj, pC_EC):
    #print("eqGDP")
    
    mask = np.full(len(pCj), True)
    mask[E] = False

    zero= -1 + GDP / ( np.sum( pCj*(Cj+Gj+Ij), where=mask)+ Cj[E] * pC_EC + sum(pXj*Xj-pMj*Mj) )

    return zero



#tp=time_previous
def eqGDPPI(GDPPI,pCj,pCtp,pXj,pXtp,Cj,Gj,Ij,Xj,Mj,Ctp,Gtp,Itp,Xtp,Mtp):

   

    zero=-1 + GDPPI / sqrt(
        ( sum( pCj*(Cj+Gj+Ij)+pXj*(Xj-Mj) ) / sum( pCtp*(Cj+Gj+Ij)+pXtp*(Xj-Mj) ) ) * ( sum( pCj*(Ctp+Gtp+Itp)+pXj*(Xtp-Mtp) ) / sum( pCtp*(Ctp+Gtp+Itp)+pXtp*(Xtp-Mtp) ) )
    )

    return zero

def eqCPI(CPI,pCj,pCtp,Cj,Ctp):

   

    zero=-1 + CPI / sqrt(
        ( sum( pCj*Cj ) / sum( pCtp*Cj ) ) * ( sum( pCj*Ctp ) / sum( pCtp*Ctp ) )
    )

    return zero

#GDPPI is the GDPPI time series
def eqGDPreal(GDPreal, GDP, GDPPI):
   
    zero=-1 + GDPreal /(
        GDP / np.prod(GDPPI)
    )
    return zero

def eqRreal(Rreal, R, CPI):
   
    zero=-1 + Rreal /(
        R / np.prod(CPI)
    )
    return zero



#I put the if  otherwise indexing with None gave me an array of array TODO
def eqCalibi(pX, Xj, data, _index = None):
    #print("eqCalibi")
    if isinstance(_index, np.ndarray):
        zero = -1 + data[_index] / (pX*Xj)[_index] #QUI
    else:
        zero = -1 + data / (pX*Xj)

    return zero


def eqCalibij(pYi, Yij, data, _index=None):
    #print("eqCalibij")
    pYid = np.diag(pYi)
    
    if isinstance(_index, np.ndarray):
        zero = -1 + data[_index[0],_index[1]] / np.dot(pYid,Yij)[_index[0],_index[1]]#QUI

    else:
        zero = -1 + data / np.dot(pYid,Yij) #QUI
        zero=zero.flatten()
        
    return zero


def eqCETquantity(Xj,Yj,alphaXj,pXj,pYj,sigmaj):

    zero = -1 + Xj / (
        np.float_power(alphaXj*pYj/pXj, sigmaj)*Yj
    )

    return zero

def eqsD(sD,Ij,pCj, Mj, Xj, pXj, GDP):
    zero = -1 + sD/(
        sum(Ij*pCj+(Xj-Mj)*pXj)/GDP
        )
    return zero

def eqT(T,tauYj, pYj, Yj, tauSj, pSj, Sj, tauL, w, Lj):

    zero = - 1 + T / (
        
        sum(   tauSj*pSj*Sj + (tauYj/(tauYj+1))*pYj*Yj + tauL*w*Lj  )
        
        )
    return zero

def eqPriceTax(pGross,pNet, tau, exclude_idx=None):
    idx=np.array(range(len(pNet)))
    mask = ~np.isin(idx, exclude_idx)
    idx = idx[mask]
    
    zero = - 1 + (pGross / (
        pNet*(1+tau)
        ))[idx]
    return zero

def eqRi(Ri,sL,w,Lj,sK,Kj,pK,sG,T,Rg,B):

    zero = - 1 + Ri / ( sL*w*sum(Lj) + sK*sum(Kj)*pK + sG*(T-Rg) - B )
    return zero

def eqIneok(I,K,alphaIK):
    
    zero = -1 + I/(alphaIK*K)
    
    return zero


def equ(u,L,Lj):
    zero = - 1 + u / (( L - sum(Lj) ) / L) 
    return zero

def eqw_real(w_real,CPI,w):
    zero=-1 + w_real /(
        w / np.prod(CPI)
    )
    return zero

def eqw_curve(w_real, alphaw, u, sigmaw):
    zero = 1 - w_real / ( alphaw *(u**sigmaw) )
    
    return zero

def eqIj(Ij,alphaIj,I,_index=None):
    
    if isinstance(_index, np.ndarray):
        zero= -1 + Ij[_index] / ( alphaIj[_index] * I)
    else:
        zero= -1 + Ij / ( alphaIj * I )
    
    return zero

def eqpI(pI,pCj,alphaIj):
    zero = 1 - pI / ( sum(pCj*alphaIj) )
    return zero
    
def eqinventory(Knext,K,delta, I):
    zero = 1 - Knext / ( K * (1-delta) + I )
    return zero
    

# _____ _   _ _____ ____   ______   __   ____ ___  _   _ ____  _     ___ _   _  ____ 
#| ____| \ | | ____|  _ \ / ___\ \ / /  / ___/ _ \| | | |  _ \| |   |_ _| \ | |/ ___|
#|  _| |  \| |  _| | |_) | |  _ \ V /  | |  | | | | | | | |_) | |    | ||  \| | |  _ 
#| |___| |\  | |___|  _ <| |_| | | |   | |__| |_| | |_| |  __/| |___ | || |\  | |_| |
#|_____|_| \_|_____|_| \_\\____| |_|    \____\___/ \___/|_|   |_____|___|_| \_|\____|
#                                                                                    


def eqsum_arr(tot, *args):
    # Check if all arguments are NumPy arrays and have the same shape
    if all(isinstance(arg, np.ndarray) for arg in args) and all(arg.shape == args[0].shape for arg in args[1:]):
        # If arguments are arrays of the same shape, perform element-wise sum
        result = np.sum(args, axis=0)
    else:
        # If arguments are not arrays of the same shape, print an error message and stop the program
        raise ValueError("Error: Arguments must be NumPy arrays of the same shape.")
    zero = 1 - tot / result  
    
    return zero


def eqsum_scalar(tot, *args):
    this_sum = sum(np.sum(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
    zero = 1 - tot/this_sum
    return zero


def eqsum_pEYE(p_CE, pY_Ej, C_E, Y_Ej, pE_B, C_EB, YE_Bj, pE_Pj, YE_Pj, pE_TnT, pE_TT, C_ET, YE_Tj, pE_Ej, YE_Ej):
    p_Ej=np.append(pY_Ej,p_CE)
    Q_Ej=np.append(Y_Ej,C_E)
    Q_EB=np.append(YE_Bj,C_EB)
    Q_EP=np.append(YE_Pj,0)
    Q_ET=np.append(YE_Tj,C_ET)
    Q_EE=np.append(YE_Ej,0)
    pE_Pj=np.append(pE_Pj,0)
    pE_Tj=np.array([pE_TnT]*(len(Q_Ej)))
    pE_Tj[T]=pE_TT
    pE_Ej = np.append(pE_Ej,0)
    zero = 1 - p_Ej*Q_Ej / (pE_B * Q_EB + pE_Tj * Q_ET + pE_Pj * Q_EP + pE_Ej * Q_EE)
    return zero

def eqsum_pESE(p_SE, tauSE, S_E,Y_Ej,C_E, pY_Ej, p_CE):
    p_E = np.append(pY_Ej,p_CE)
    Q_E = np.append(Y_Ej,C_E)
    zero = 1 - p_SE*S_E*(1+tauSE) / sum((p_E*Q_E))
    return zero

def compute_new_E(n_sectors, idx_E, _index):
    zero_indexes = [i for i in range(n_sectors) if i not in _index]
    E_diff = sum(1 for el in zero_indexes if el < idx_E)
    new_E = idx_E - E_diff
    return new_E

def eqCobbDouglasj_lambda(Qj,alphaQj,pCj,Q, lambda_E, lambda_nE, _index=None):
    p_CE=pCj[E]
    C_E =  lambda_E * alphaQj[E] * (Q/ p_CE) 
    
    if isinstance(_index, np.ndarray):
        new_E=compute_new_E(n_sectors=len(pCj), idx_E=E, _index=_index)
        
        Cj= lambda_nE * alphaQj[_index] * (Q/ pCj[_index])
        Cj[new_E]=C_E
        zero= -1 + Qj[_index] / Cj
        
    else:
        Cj= lambda_nE * alphaQj * (Q/ pCj)
        Cj[E]=C_E        
        
        zero= -1 + Qj / Cj
    
    return zero



def eqlambda_nE(alphaCj,lambda_E, lambda_nE):
    
    sum_alpha_Cj = sum(value for index, value in enumerate(alphaCj) if index != E)
    
    zero = 1 -  1/ (lambda_nE * sum_alpha_Cj + lambda_E * alphaCj[E])
    
    return zero




# def GPI(GDPPI,pCj,pCtp,pXj,pXtp,Cj,Gj,Ij,Xj,Mj,Yij,Ctp,Gtp,Itp,Xtp,Mtp,Yijtp,idxC=len(Cj),idxG,idxI,idxX)

# def eqCPI(GDPPI,pCj,pCtp,Cj,Gj,Ij,Ctp,Gtp,Itp):

#     GDPPI=GDPPI.item()

#     zero=-1 + GDPPI / sqrt(
#         ( sum( pCj*(Cj+Gj+Ij)+pXj*(Xj-Mj) ) / sum( pCtp*(Cj+Gj+Ij)+pXtp*(Xj-Mj) ) ) * ( sum( pCj*(Ctp+Gtp+Itp)+pXj*(Xtp-Mtp) ) / sum( pCtp*(Ctp+Gtp+Itp)+pXtp*(Xtp-Mtp) ) )
#     )

#     return zero


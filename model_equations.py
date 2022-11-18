#equations cge
import numpy as np
from math import sqrt
import data_calibration_from_matrix as dt

#EQUATIONS

def eqKLj(KLj,bKL, bKLj, Lj, Kj, alphaLj, alphaKj):
    #print("eqKLj")

    zero = -1 + KLj / (
        bKL*bKLj*np.float_power(Lj,alphaLj) * np.float_power(Kj,alphaKj)
    )

    return zero

# same equation for Lj and Kj (Factors)

def eqFj(Fj,pF,KLj,pKLj,alphaFj):
    pF=pF.item()

    zero= -1 + Fj / (
        (np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0))/pF
    )

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


def eqpYj(pYj,pSj,aKLj,pKLj,aYij):

    pSjd=np.diag(pSj)

    zero= -1 + pYj / (
        ( aKLj * pKLj + np.dot(pSjd,aYij).sum(axis=0) ) #AXIS=1 sum over the rows CHECKED
    )

    return zero


def eqCET(Zj, alphaXj,alphaYj,Xj,Yj,sigmaj):

    etaj = (sigmaj-1)/sigmaj

    partj = alphaXj * np.float_power(Xj,etaj, out=np.zeros(len(Xj)),where=(Xj!=0)) + alphaYj * np.float_power(Yj,etaj)

    zero = -1 + Zj / np.float_power(partj, 1/etaj)

    return zero


def eqCESquantity(Xj, Zj, alphaXj, alphaYj, pXj, pYj, sigmaj, _index=None):
        
    if isinstance(_index, np.ndarray):
        Xj=Xj[_index]
        Zj=Zj[_index]
        alphaXj=alphaXj[_index]
        alphaYj=alphaYj[_index]
        pXj=pXj[_index]
        pYj=pYj[_index]
        sigmaj=sigmaj[_index]
        
    
    #is it correct??? TODO
    partj = np.float_power(alphaXj,sigmaj, out=np.zeros(len(alphaXj)),where=(alphaXj!=0)) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj, out=np.zeros(len(alphaYj)),where=(alphaYj!=0))* np.float_power(pYj,1-sigmaj)
    
    zero= -1 + Xj / (
        np.float_power(alphaXj/pXj, sigmaj) * np.float_power(partj , sigmaj/(1-sigmaj) ) * Zj
    )
    return zero


def eqCESprice(pZj,pXj,pYj,alphaXj,alphaYj,sigmaj):

    partj= np.float_power(alphaXj,sigmaj, out=np.zeros(len(alphaXj)),where=(alphaXj!=0)) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj, out=np.zeros(len(alphaYj)),where=(alphaYj!=0)) * np.float_power(pYj,1-sigmaj)

    zero= -1 + pZj / (
        np.float_power(partj, 1/(1-sigmaj))
    )

    return zero


def eqB(B,pXj,Xj,pMj,Mj):
    
    zero = -1 + B / sum(pXj*Xj-pMj*Mj)
    
    return zero


def eqwB(X,wX,GDP):
    
    zero = -1 + X / (wX*GDP)
    
    return(zero)

##check for the case where the index is an empty array! TODO
def eqCj(Cj,alphaCj,pCj,R,_index=None):
    
    if isinstance(_index, np.ndarray):
        zero= -1 + Cj[_index] / ( alphaCj[_index] * (R/ pCj[_index]) )
    else:
        zero= -1 + Cj / ( alphaCj * (R/ pCj) )
    
    return zero

def eqalphaCj(alphaCj,R,pCj,alphaCDESj,betaRj, _index=None):
    if isinstance(_index, np.ndarray):
        zero= -1 + alphaCj[_index] / (alphaCDESj * np.float_power( R/pCj , betaRj ) / sum(alphaCDESj * np.float_power( R/pCj , betaRj )))[_index]
    else:
        zero= -1 + alphaCj / (alphaCDESj * np.float_power( R/pCj , betaRj ) / sum(alphaCDESj * np.float_power( R/pCj , betaRj )))
    
    return zero


def eqR(R,Cj,pCj):

    zero = -1 + R / sum(Cj*pCj)

    return zero


def eqw(pXj,Xj,wXj,GDP, _index = None):
    
    if isinstance(_index, np.ndarray):
        deno = (pXj * Xj)[_index]
        nume = (wXj * GDP)[_index]
    else:
        deno = (pXj * Xj)
        nume = (wXj * GDP)
    
    zero = -1 + nume / deno

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

    F=F.item()

    zero= -1 + F / sum(Fj)

    return zero


def eqID(x,y):
    #print("eqID")
    zero=-1 + x / y

    return zero


def eqGDP(GDP,pCj,Cj,Gj,Ij,pXj,Xj,pMj,Mj):
    #print("eqGDP")
    GDP=GDP.item()

    zero= -1 + GDP / sum(pCj*(Cj+Gj+Ij)+pXj*Xj-pMj*Mj)

    return zero


#tp=time_previous
def eqGDPPI(GDPPI,pCj,pCtp,pXj,pXtp,Cj,Gj,Ij,Xj,Mj,Ctp,Gtp,Itp,Xtp,Mtp):

    GDPPI=GDPPI.item()

    zero=-1 + GDPPI / sqrt(
        ( sum( pCj*(Cj+Gj+Ij)+pXj*(Xj-Mj) ) / sum( pCtp*(Cj+Gj+Ij)+pXtp*(Xj-Mj) ) ) * ( sum( pCj*(Ctp+Gtp+Itp)+pXj*(Xtp-Mtp) ) / sum( pCtp*(Ctp+Gtp+Itp)+pXtp*(Xtp-Mtp) ) )
    )

    return zero

def eqCPI(CPI,pCj,pCtp,Cj,Ctp):

    CPI=CPI.item()

    zero=-1 + CPI / sqrt(
        ( sum( pCj*Cj ) / sum( pCtp*Cj ) ) * ( sum( pCj*Ctp ) / sum( pCtp*Ctp ) )
    )

    return zero

#GDPPI is the GDPPI time series
def eqGDPreal(GDPreal, GDP, GDPPI):
    GDP.item()
    zero=-1 + GDPreal /(
        GDP / np.prod(GDPPI)
    )
    return zero

def eqRreal(Rreal, R, CPI):
    R.item()
    zero=-1 + Rreal /(
        R / np.prod(CPI)
    )
    return zero



#I put the if  otherwise indexing with None gave me an array of array TODO
def eqCalibi(pX, Xj, data, _index = None):
    #print("eqCalibi")
    if isinstance(_index, np.ndarray):
        
        zero = -1 + data[_index] / (pX*Xj)[_index]#QUI

    else:

        zero = -1 + data / (pX*Xj)
        

    return zero


def eqCalibij(pYi, Yij, data, _index=None):
    #print("eqCalibij")
    pYid = np.diag(pYi)
    
    if isinstance(_index, np.ndarray):
        zero = -1 + data[_index[0],_index[1]] / np.dot(pYid,Yij)[_index[0],_index[1]]#QUI

    else:
        zero = -1 + data / np.dot(pYid,Yij)#QUI
        zero=zero.flatten()
        
    return zero


def eqCETquantity(Xj,Yj,alphaXj,pXj,pYj,sigmaj):

    zero = -1 + Xj / (
        np.float_power(alphaXj*pYj/pXj, sigmaj)*Yj
    )

    return zero

# def eqCPI(GDPPI,pCj,pCtp,Cj,Gj,Ij,Ctp,Gtp,Itp):

#     GDPPI=GDPPI.item()

#     zero=-1 + GDPPI / sqrt(
#         ( sum( pCj*(Cj+Gj+Ij)+pXj*(Xj-Mj) ) / sum( pCtp*(Cj+Gj+Ij)+pXtp*(Xj-Mj) ) ) * ( sum( pCj*(Ctp+Gtp+Itp)+pXj*(Xtp-Mtp) ) / sum( pCtp*(Ctp+Gtp+Itp)+pXtp*(Xtp-Mtp) ) )
#     )

#     return zero


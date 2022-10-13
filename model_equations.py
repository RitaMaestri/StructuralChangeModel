#equations cge
import numpy as np
from math import sqrt



#EQUATIONS

def eqKLj(KLj, bKL, Lj, Kj, alphaLj, alphaKj):
    #print("eqKLj")

    zero = KLj - bKL * np.float_power(Lj,alphaLj) * np.float_power(Kj,alphaKj)
    
    return zero 

# same equation for Lj and Kj (Factors)

def eqFj(Fj,pF,KLj,pKLj,alphaFj):   
    #print("eqFj")
    pF=pF.item()
    
    zero= Fj-(np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0))/pF
    
    return zero


def eqYij(Yij,aYij,Yj):
    #print("eqYij")

    Yjd=np.diag(Yj)
    
    zero=Yij-np.dot(aYij,Yjd)
    
    #convert matrix to vector
    zero=zero.flatten()
    return zero


def eqKL(KLj,aKLj,Yj):
    #print("eqKL")
    zero=KLj- np.multiply(aKLj,Yj)
    
    return zero


def eqpYj(pYj,pSj,aKLj,pKLj,aYij):

    pSjd=np.diag(pSj)
    
    zero= pYj - ( aKLj * pKLj + np.dot(pSjd,aYij).sum(axis=0) ) #AXIS=1 sum over the rows CHECKED
    
    return zero


def eqCET(Zj, alphaXj,alphaYj,Xj,Yj,sigmaj):
    
    etaj = (sigmaj-1)/sigmaj 
    
    partj = alphaXj * np.float_power(Xj,etaj) + alphaYj * np.float_power(Yj,etaj)
    
    zero = Zj - np.float_power(partj,(1/etaj))  
    
    return zero


def eqCESquantity(Xj, Zj, alphaXj, alphaYj, pXj, pYj, sigmaj):
    
    #etaj=(sigmaj-1)/sigmaj 
    
    partj = np.float_power(alphaXj,sigmaj) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj) * np.float_power(pYj,1-sigmaj)

    zero= Xj - np.float_power(alphaXj/pXj, sigmaj) * np.float_power(partj , sigmaj/(1-sigmaj) ) * Zj
    
    return zero


def eqCESprice(pZj,pXj,pYj,alphaXj,alphaYj,sigmaj):
    
    partj= np.float_power(alphaXj,sigmaj) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj) * np.float_power(pYj,1-sigmaj)

    zero= pZj - np.float_power(partj, 1/(1-sigmaj))
    
    return zero
    

def eqB(B,pXj,Xj,pMj,Mj):
    
    zero = B - sum(pXj*Xj-pMj*Mj)
    
    return zero


def eqwB(X,wX,GDP):
    
    zero = X - wX*GDP
    
    return(zero)


def eqCj(Cj,alphaCj,pCj,R):         

    zero= Cj - alphaCj * (R/ pCj)
    
    return zero

def eqR(R,Cj,pCj):
    
    zero = R - sum(Cj*pCj)
    
    return zero


def eqw(pXj,Xj,wXj,GDP):
    
    zero = pXj * Xj - wXj * GDP
    
    return zero


def eqSj(Sj,Cj,Gj,Ij,Yij): 
    #print("eqSj")
    zero = Sj - (Cj + Gj + Ij + Yij.sum(axis=1))#sum over the rows 
    
    return zero


#same equation for L and K

def eqF(F,Fj):
    #print("eqF")
    
    F=F.item()
    
    zero= F-sum(Fj)
    
    return zero


def eqID(x,y):
    #print("eqID")
    zero=x-y
    
    return zero



def eqGDP(GDP,pCj,Cj,Gj,Ij,pXj,Xj,pMj,Mj):
    #print("eqGDP")
    GDP=GDP.item()
    
    zero= GDP - sum(pCj*(Cj+Gj+Ij)+pXj*Xj-pMj*Mj)
    
    return zero



#tp=time_previous
def eqGDPPI(GDPPI,pCj,pCtp,Cj,Gj,Ij,Ctp,Gtp,Itp):
    
    GDPPI=GDPPI.item()
    
    zero=GDPPI - sqrt( ( sum( pCj*(Cj+Gj+Ij) ) / sum( pCtp*(Cj+Gj+Ij) ) ) * ( sum( pCj*(Ctp+Gtp+Itp) ) / sum( pCtp*(Ctp+Gtp+Itp) ) ) )
    
    return zero

#GDPPI is the GDPPI time series
def eqGDPreal(GDPreal, GDP, GDPPI):
    GDP.item()
    zero=GDPreal - GDP / np.prod(GDPPI)
    return zero



def eqCalibi(pX, Xj, data):
    #print("eqCalibi")
    zero = data - pX*Xj
    
    return zero

def eqCalibij(pYi, Yij, data):
    #print("eqCalibij")
    pYid = np.diag(pYi)
    
    zero = data - np.dot(pYid,Yij)
    
    zero=zero.flatten()
    
    return zero


def eqCETquantity(Xj,Yj,alphaXj,pXj,pYj,sigmaj):
    
    zero = Xj - np.float_power(alphaXj*pYj/pXj, sigmaj)*Yj
    
    return zero


def eqGDPreduced(GDP,pCj,Cj,pMj,Mj):
    #print("eqGDP")
    GDP=GDP.item()
    
    zero= GDP - sum(pCj*Cj-pMj*Mj)
    
    return zero

#equations cge
import numpy as np
from math import sqrt

# def shapeError(*args):
#     if not all(x.shape[0] == args[0].shape[0] for x in args):
#         raise Exception("Wrong shape")


# def notScalarError(*args):
#     if not all(np.isscalar(x) for x in args):
#         raise Exception("Scalar needed")


# def notArrayError(*args):

#     if not all(isinstance(i, np.ndarray) for i in args):
#         raise Exception("Array needed")


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
    #print("eqpYj")
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


def eqw(B,wB,GDP):
    
    zero = B - wB*GDP
    
    return(zero)


def eqCj(Cj,alphaCj,pCj,R):         #eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],R=d['R']),
    #print("eqCj")
    #pL,pK=pL.item(),pK.item()

    zero= Cj - alphaCj * (R/ pCj)
    
    return zero

# def eqCj(Cj,alphaCj,pCj,pL,Lj,pK,Kj):         #eq.eqCj(Cj=d['Cj'],alphaCj=d['alphaCj'],pCj=d['pCj'],pL=d['pL'],Lj=d['Lj'],pK=d['pK'],Kj=d['Kj']),

#     #print("eqCj")
#     pL,pK=pL.item(),pK.item()

#     zero= Cj - alphaCj * ((pL*sum(Lj)+pK*sum(Kj))/ pCj)
    
#     return zero
    
    
def eqSj(Sj,Cj,Yij):
    #print("eqSj")
    zero = Sj - (Cj + Yij.sum(axis=1))#sum over the rows
    
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

def eqGDP(GDP,pCj,Cj,pXj,Xj,pMj,Mj):
    #print("eqGDP")
    GDP=GDP.item()
    
    zero= GDP - sum(pCj*Cj+pXj*Xj-pMj*Mj)
    
    return zero



#tp=time_previous
def eqGDPPI(GDPPI,pCj,pCtp,Cj, Ctp):
    
    GDPPI=GDPPI.item()
    
    zero=GDPPI - sqrt( ( sum(pCj*Cj)/sum(pCtp*Cj) ) * ( sum(pCj*Ctp)/sum(pCtp*Ctp) ) )
    
    return zero

#GDPPI is the GDPPI time series
def eqGDPreal(GDPreal, GDP, GDPPI):
    GDP.item()
    zero=GDPreal - GDP / np.prod(GDPPI)
    return zero

def eqCalibi(pX, Xj, data):
    #print("eqCalibi")
    zero=data - pX*Xj
    
    return zero

def eqCalibij(pYi, Yij, data):
    #print("eqCalibij")
    pYid=np.diag(pYi)
    
    zero = data - np.dot(pYid,Yij)
    
    zero=zero.flatten()
    return zero


# def eqpKLj(pKLj,Lj,pL,Kj,pK):
    
#     zero = pKLj - (pL*Lj + pK*Kj)
    
#     return zero




def eqCETquantity(Xj,Yj,alphaXj,pXj,pYj,sigmaj):
    
    zero = Xj - np.float_power(alphaXj*pYj/pXj, sigmaj)*Yj
    
    return zero


def eqGDPreduced(GDP,pCj,Cj,pMj,Mj):
    #print("eqGDP")
    GDP=GDP.item()
    
    zero= GDP - sum(pCj*Cj-pMj*Mj)
    
    return zero

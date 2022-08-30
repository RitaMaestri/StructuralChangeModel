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

def eqKLj(KLj, bKLj, Lj, Kj, alphaLj, alphaKj):
    #print("eqKLj")

    zero = KLj - bKLj * np.float_power(Lj,alphaLj) * np.float_power(Kj,alphaKj)
    
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


def eqpYj(pYj,aKLj,pKLj,aYij):
    #print("eqpYj")
    pYjd=np.diag(pYj)
    
    zero=pYj - (np.multiply(aKLj,pKLj) + np.dot(pYjd,aYij).sum(axis=0))
    
    return zero


def eqCj(Cj,alphaCj,pCj,pL,Lj,pK,Kj):
    #print("eqCj")
    pL,pK=pL.item(),pK.item()

    zero= Cj - alphaCj * ((pL*sum(Lj)+pK*sum(Kj))/ pCj)
    
    return zero
    
    
def eqYj(Yj,Cj,Yij):
    #print("eqYj")
    zero = Yj-(Cj + Yij.sum(axis=1))
    
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

def eqGDP(GDP,pCj,Cj):
    #print("eqGDP")
    GDP=GDP.item()
    zero= GDP - sum(pCj*Cj)
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


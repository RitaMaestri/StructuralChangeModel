#equations cge
import numpy as np

def eqKLj(KLj, bKLj, Lj, Kj, alphaLj, alphaKj):

    zero = KLj - bKLj * np.power(Lj,alphaLj) * np.power(Kj,alphaKj)
    
    return zero

#same equation for Lj and Kj (Factors) Fbar is the complementary factor (K if F=L and viceversa)

def eqFj(Fj,GDP,pFbar,Fbar,F,KLj,pKLj,alphaFj):
    
    zero= Fj-(np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0)*F)/(GDP-pFbar*Fbar)
    
    return zero


def eqYij(Yij,aYij,Yj):
    
    Yjd=np.diag(Yj)
    
    zero=Yij-np.dot(aYij,Yjd)
    #convert matrix to vector
    zero=zero.flatten()
    
    return zero



def eqKL(KLj,aKLj,Yj):
    
    zero=KLj-np.multiply(aKLj,Yj)
    
    return zero



def eqpYj(pYj,aKLj,pKLj,aYij):
    
    pYjd=np.diag(pYj)

    zero=pYj - (np.multiply(aKLj,pKLj) + np.dot(pYjd,aYij).sum(axis=0))
    
    return zero


def eqCj(Cj,alphaCj,pCj,GDP):
    
    zero=Cj-alphaCj*GDP/pCj
    
    return zero
    

    
def eqYj(Yj,Cj,Yij):
    
    zero = Yj-(Cj + Yij.sum(axis=1))
    
    return zero


#same equation for L and K
def eqF(F,Fj):
    
    zero= F-sum(Fj)
    
    return zero


def eqpCj(pCj,pYj):
    
    zero=pCj-pYj
    
    return zero

def eqGDP(GDP,pL,L,pK,K):
    
    zero= GDP-( pL*L+ pK*K)
    
    return zero
    
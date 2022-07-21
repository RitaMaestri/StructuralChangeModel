#equations cge
import numpy as np

def shapeError(*args):
    if not all(x.shape[0] == args[0].shape[0] for x in args):
        raise Exception("Wrong shape")


def notScalarError(*args):
    if not all(np.isscalar(x) for x in args):
        raise Exception("Scalar needed")


def notArrayError(*args):

    if not all(isinstance(i, np.ndarray) for i in args):
        raise Exception("Array needed")


#EQUATIONS

def eqKLj(KLj, bKL, Lj, Kj, alphaLj, alphaKj):
    #print(bKL)    notArrayError(KLj, bKL, Lj, Kj, alphaLj, alphaKj)
    shapeError(KLj, Lj, Kj, alphaLj, alphaKj)
    bKL=bKL.item()

    zero = KLj - bKL * np.power(Lj,alphaLj) * np.power(Kj,alphaKj)
    
    return zero 

# same equation for Lj and Kj (Factors)

def eqFj(Fj,pF,KLj,pKLj,alphaFj):
    
   # type and shape checks
    
    notArrayError(Fj,pF,KLj,pKLj,alphaFj)
    pF=pF.item()
    notScalarError(pF)
    shapeError(Fj,KLj,pKLj,alphaFj)
    
    # final result
    
    zero= Fj-(np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0))/pF
    
    return zero


def eqYij(Yij,aYij,Yj):
    notArrayError(Yij,aYij,Yj)
    shapeError(Yij,aYij,Yj)
    
    Yjd=np.diag(Yj)
    
    zero=Yij-np.dot(aYij,Yjd)
    
    #convert matrix to vector
    zero=zero.flatten()
    
    return zero


def eqKL(KLj,aKLj,Yj):
    notArrayError(KLj,aKLj,Yj)
    shapeError(KLj,aKLj,Yj)
    
    zero=KLj- np.multiply(aKLj,Yj)
    
    return zero


def eqpYj(pYj,aKLj,pKLj,aYij):
    notArrayError(pYj,aKLj,pKLj,aYij)
    shapeError(pYj,aKLj,pKLj,aYij)
    
    pYjd=np.diag(pYj)
    
    zero=pYj - (np.multiply(aKLj,pKLj) + np.dot(pYjd,aYij).sum(axis=0))
    
    return zero


def eqCj(Cj,alphaCj,pCj,pL,L,pK,K):
    notArrayError(Cj,alphaCj,pCj,pL,L,pK,K)
    shapeError(Cj,alphaCj,pCj)
    
    L,pL,K,pK=L.item(),pL.item(),K.item(),pK.item()

    notScalarError(L,pL,K,pK)
    #("Cj=", Cj, "alphaCj= ", alphaCj, "pL*L+pK*K=", pL*L+pK*K, "pCj", pCj  )
    zero= Cj - alphaCj * (pL*L+pK*K) / pCj
    
    return zero
    
    
def eqYj(Yj,Cj,Yij):
    notArrayError(Yj,Cj,Yij)
    shapeError(Yj,Cj,Yij)
    
    zero = Yj-(Cj + Yij.sum(axis=1))
    
    return zero


#same equation for L and K

def eqF(F,Fj):
    notArrayError(F,Fj)

    F=F.item()
    #print("F=", F, "Fj=",Fj, " sum(Fj)=", sum(Fj))
    zero= F-sum(Fj)
    
    return zero


def eqpCj(pCj,pYj):
    notArrayError(pCj,pYj)
    shapeError(pCj,pYj)
     
    zero=pCj-pYj
    
    return zero

def eqGDP(GDP,pL,L,pK,K):
    GDP=GDP.item()
    pL=pL.item()
    L=L.item()
    pK=pK.item()
    K=K.item()
    
    zero= GDP-( pL*L+ pK*K)
    
    return zero
    
#equations cge
import numpy as np

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
    
    bKL=bKL.item()

    zero = KLj - bKL * np.power(Lj,alphaLj) * np.power(Kj,alphaKj)
    
    return zero 

# same equation for Lj and Kj (Factors)

def eqFj(Fj,pF,KLj,pKLj,alphaFj):   

    pF=pF.item()
    
    zero= Fj-(np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0))/pF
    
    return zero


def eqYij(Yij,aYij,Yj):
    
    Yjd=np.diag(Yj)
    
    zero=Yij-np.dot(aYij,Yjd)
    
    #convert matrix to vector
    zero=zero.flatten()
    
    return zero


def eqKL(KLj,aKLj,Yj):
    
    zero=KLj- np.multiply(aKLj,Yj)
    
    return zero


def eqpYj(pYj,aKLj,pKLj,aYij):
    
    pYjd=np.diag(pYj)
    
    zero=pYj - (np.multiply(aKLj,pKLj) + np.dot(pYjd,aYij).sum(axis=0))
    
    return zero


def eqCj(Cj,alphaCj,pCj,pL,Lj,pK,Kj):
    
    pL,pK=pL.item(),pK.item()

    zero= Cj - alphaCj * (pL*sum(Lj)+pK*sum(Kj)) / pCj
    
    return zero
    
    
def eqYj(Yj,Cj,Yij):
    
    zero = Yj-(Cj + Yij.sum(axis=1))
    
    return zero


#same equation for L and K

def eqF(F,Fj):
    F=F.item()
    
    zero= F-sum(Fj)
    
    return zero


def eqpCj(pCj,pYj):

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
    
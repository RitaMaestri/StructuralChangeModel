import pandas as pd

import numpy as np

import random

random.seed(10)

N=2


Mmin=100
Mmax=900

#I have to remove the K,L columns and the H row since the many zeros prevent the algorithm from converging

M=np.random.randint(Mmin,Mmax,size=(N+2,N+1))

M[-1][-1]=0
M[-2][-1]=0


rowsum=np.random.randint(Mmin,Mmax,size=N+2)
colsum=np.append(rowsum[:-2],rowsum[-1]+rowsum[-2])
rowsum = np.array([rowsum]).T

while (not (np.all(abs(rowsum.T-M.sum(axis=1))<1e-15) and np.all(abs(colsum-M.sum(axis=0))<1e-10))):
        
    colsummat=np.tile(colsum,(N+2,1))
    Mmult = np.multiply(M, colsummat)
    dividendcol = np.tile((M.sum(axis=0)),(N+2,1))
    M=np.divide(Mmult,dividendcol)   
    
    # colsummat=np.tile(colsum,(N+2,1))
    # Mmult = np.multiply(M, colsummat)
    # dividendcol = np.tile((M.sum(axis=0)),(N+2,1))
    # M=np.divide(Mmult,dividendcol)
    
    rowsummat=np.tile(rowsum,(1,N+1))
    Mmult = np.multiply(rowsummat,M)
    dividendrow= np.tile(np.array([M.sum(axis=1)]).T,(1,N+1))
    M=np.divide(Mmult,dividendrow)



print("rowsum",rowsum.T==M.sum(axis=1))
print("colsum",colsum==M.sum(axis=0))

#append final demand row and C, K columns
M = np.vstack(( M, np.zeros((1,M.shape[1])) ))
M = np.hstack((M[:,:N], np.zeros((M.shape[0],2)) , M[:,N:]))

#add sum of K and L
M[N+2,N]=sum(M[N,:])
M[N+2,N+1]=sum(M[N+1,:])

#test
M.sum(axis=1)==M.sum(axis=0)

#to Data Frame
names=list(map(str, range(1,N+1)))
names.extend("KLH")
M=pd.DataFrame(M, names, names)

#export matrix
M.to_csv(path_or_buf="/home/rita/Documents/Stage/Code/Matrix "+str(N)+"x"+str(N)+".csv")

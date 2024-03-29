"""
Created on Tue Apr 18 18:12:28 2023

@author: rita
"""

import numpy as np
import pandas as pd
import glob
import sys


mydict={}


region="EUR"
#folder_all="/home/rita/Documents/Tesi/Code/aggregation GTAP/SAMfiles/all/"
folder_all="/home/rita/Documents/Tesi/Code/aggregation GTAP/SAMfiles EUR 3/SAMfiles/all/"
#folder_reg="/home/rita/Documents/Tesi/Code/aggregation GTAP/SAMfiles/france/"    
folder_reg="/home/rita/Documents/Tesi/Code/aggregation GTAP/SAMfiles EUR 3/SAMfiles/EUR/"


files_all = glob.glob(folder_all+"*.csv")
#create a dictionary whose keys are the name of the files 
#in the folder mentioned above and whose value is the value for FRANCE
for f in files_all:
    df = pd.read_csv(f,sep="|",index_col=0)
    key = f.replace(folder_all,"").replace(".csv", "")
    key = f.replace(folder_all,"").replace(".csv", "")
    mydict[key]=df.loc[region]

files_reg = glob.glob(folder_reg+"*.csv")
#create a dictionary whose keys are the name of the files 
#in the folder mentioned above and whose value is the value of the entire df (in the folder france)
for f in files_reg:
    df = pd.read_csv(f,sep="|",index_col=0)
    key = f.replace("_"+region+".csv", "").replace(folder_reg,"")
    mydict[key]=df

#these correspondences are taken from the file aggregated_inputoutput_table_FRA

N=len(mydict["Capital"])
sectors=list(mydict["Capital"].index.values)


production_taxes = np.array(mydict["T_prod"])

sales_taxes = np.array(mydict["T_CI_dom_cor"].sum(axis=0) +
               mydict["T_CI_imp_cor"].sum(axis=0) + 
               
               mydict["T_FBCF_dom_cor"] + 
               mydict["T_FBCF_imp_cor"] + 
               
               mydict["T_Hsld_dom_cor"] + 
               mydict["T_Hsld_imp_cor"] + 
               
               mydict["T_AP_dom_cor"] + 
               mydict["T_AP_imp_cor"] + 
               
               mydict["T_Imp_cor"] + 
               mydict["T_Exp_cor"] + 
               mydict["Auto_TMX"])

wLj =np.array(  mydict["tech_aspros"] + mydict["clerks"] + mydict["service_shop"] + mydict["off_mgr_pros"] + mydict["ag_othlowsk"] )

labor_taxes =np.array(  mydict["T_tech_aspros"] + mydict["T_clerks"] + mydict["T_service_shop"] + mydict["T_off_mgr_pros"] + mydict["T_ag_othlowsk"] )

K = np.array( mydict["Capital"] + mydict["T_Capital"] )

R = np.array( mydict["NatRes"] + mydict["Land"] + mydict["T_Land"] + mydict["T_NatRes"] )

pMjMj = np.array( mydict["Imp_cor"] + mydict["Imp_trans_cor"] )

pCiYij = np.array( mydict["CI_imp_cor"] + mydict["CI_dom_cor"] )

pCjCj = np.array( mydict["C_hsld_imp"] + 	mydict["C_hsld_dom"] )

pCjGj = np.array( mydict["C_AP_imp"] + mydict["C_AP_dom"] )

pCjIj = np.array( mydict["FBCF_imp"] + mydict["FBCF_dom"] )

pXjXj= np.array( mydict["Exp_trans_cor"] + mydict["Exp_cor"] )


production_taxes[abs(production_taxes)<10e-7] = 0
sales_taxes[abs(sales_taxes)<10e-7] = 0
wLj[abs(wLj)<10e-7] = 0
labor_taxes[abs(labor_taxes)<10e-7] = 0
K[abs(K)<10e-7] = 0
R[abs(R)<10e-7] = 0
pMjMj[abs(pMjMj)<10e-7] = 0
pCiYij[abs(pCiYij)<10e-7] = 0
pCjCj[abs(pCjCj)<10e-7] = 0
pCjGj[abs(pCjGj)<10e-7] = 0
pCjIj[abs(pCjIj)<10e-7] = 0
pXjXj[abs(pXjXj)<10e-7] = 0


pLLj= wLj + labor_taxes
pKKj = K + R

#check for equilibrium

equilibrium = ( pLLj + pKKj + production_taxes + sales_taxes + pMjMj + pCiYij.sum(axis=0) - 
               (pCjCj + pCjGj + pCjIj + pXjXj + pCiYij.sum(axis=1))
               )

print("equilibrium:", sum(equilibrium**2))
if sum(equilibrium**2)>2:
    "the system is not at the equilibrium"
    sys.exit()



#CORRECTION FOR SMALL DISEQUILIBRIUM 

cons_cost_diff= (pLLj + pKKj + production_taxes + sales_taxes + pMjMj + pCiYij.sum(axis=0) - 
                 (pCjCj + pCjGj + pCjIj + pXjXj + pCiYij.sum(axis=1)))
                 
total_domestic_production=pCjCj+pCjGj+pCjIj
pCjCj=pCjCj+(cons_cost_diff)*(pCjCj/total_domestic_production)
pCjGj=pCjGj+(cons_cost_diff)*(pCjGj/total_domestic_production)
pCjIj=pCjIj+(cons_cost_diff)*(pCjIj/total_domestic_production)

diff_final= (
    pCiYij.sum(axis=0)+pLLj+pKKj+production_taxes+sales_taxes-(
    pCiYij.sum(axis=1)+pCjCj+pCjGj+pCjIj-pMjMj+pXjXj
    )
    )




#### DERIVED QUANTITIES #######

#gross domestic product
pYjYj = pCiYij.sum(axis=0) + pLLj + pKKj + production_taxes

#KL good
pKLjKLj = pKKj + pLLj 

#Armington good
pSjSj = pCjCj + pCjGj + pCjIj + pCiYij.sum(axis=1) - sales_taxes

#domestic good
pDjDj = pSjSj - pMjMj
(pSjSj - pMjMj) / (pYjYj-pXjXj)

#GTAP Import elasticities + default export elasticities (-2 for all sectors)

Armington_elasticities=pd.read_csv("data/GTAP_Armington_elasticities3.csv", index_col="commodity").squeeze()
sigmaSj=Armington_elasticities.to_numpy()

export_elasticities=pd.read_csv("data/GTAP_export_elasticities3.csv", index_col="code").squeeze()
sigmaXj=np.array(export_elasticities)

KL_elasticities=pd.read_csv("data/KL_elasticities3.csv", index_col="commodity").squeeze()
sigmaKLj=KL_elasticities.to_numpy()

non_zero_index_G=np.array(np.where(pCjGj != 0)).flatten()
non_zero_index_I=np.array(np.where(pCjIj != 0)).flatten()
non_zero_index_X=np.array(np.where(pXjXj != 0)).flatten()
non_zero_index_M=np.array(np.where(pMjMj != 0)).flatten()
non_zero_index_C=np.array(np.where(pCjCj != 0)).flatten()
non_zero_index_K=np.array(np.where(pKKj != 0)).flatten()
non_zero_index_L=np.array(np.where(pLLj != 0)).flatten()
non_zero_index_Yij=np.array(np.where(pCiYij != 0))
len(pCiYij[pCiYij != 0])

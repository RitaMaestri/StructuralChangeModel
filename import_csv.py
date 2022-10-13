import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resources_table=pd.read_csv("data/ressources.csv")
resources_table.set_index('Activity code',inplace=True)

employment_table=pd.read_csv("data/emplois.csv")
employment_table.set_index('Activity code',inplace=True)

intermediate_cons_table=pd.read_csv("data/intermediaire.csv")
intermediate_cons_table.set_index("Branche",inplace=True)

production_table=pd.read_csv("data/production.csv")
production_table.set_index('Branche',inplace=True)
production_table=production_table.transpose()

exploitation_table=pd.read_csv("data/exploitation.csv")
exploitation_table.set_index('Branche',inplace=True)
exploitation_table=exploitation_table.transpose()

def shorten(db):
    return np.array(db.drop(["PCHTR ", "PCAFAB", "TOTAL", "TOTAL "],errors='ignore'))

# def clean_data(db):
#     short_db = shorten(db)
#     norm_db= np.array([float(i)/sum(short_db) for i in short_db])
#     return np.array(short_db) + norm_db * sum(db[["PCHTR ", "PCAFAB"]])


def clean_data(db):
    return shorten(db)

# consumptions
households=employment_table["Ménages"]
households=clean_data(households)

government=employment_table["Total APU"] + employment_table["ISBLSM"]
government= clean_data(government)

investment = employment_table["FBC totale "] 
investment = clean_data(investment)

export=employment_table["Exportations de biens et de services"]
export = clean_data(export)

#costs
imports= resources_table["Importations de biens et de services"] + resources_table["Correction CAF/FAB"]
imports=clean_data(imports)

sales_taxes = resources_table["Impôts sur les produits - total -"] + resources_table["Subventions sur les produits"]
sales_taxes=clean_data(sales_taxes)


pCjCj=households
pCjGj=government
pCjIj=investment
pXjXj=export
pMjMj=imports

com_margins=resources_table["Marges commerciales"]
com_margins=shorten(com_margins)


#distribute sales taxes and commercial margins
total_domestic_production=households+government+investment+export
pCjCj=pCjCj-(sales_taxes+com_margins)*(pCjCj/total_domestic_production)
pCjGj=pCjGj-(sales_taxes+com_margins)*(pCjGj/total_domestic_production)
pCjIj=pCjIj-(sales_taxes+com_margins)*(pCjIj/total_domestic_production)
pXjXj=pXjXj-(sales_taxes+com_margins)*(pXjXj/total_domestic_production)


#production
pYjYj=resources_table["Production des produits (1)"]
pYjYj=clean_data(pYjYj)

pLLj=exploitation_table["Rémunération des salariés"]
pLLj=shorten(pLLj)

pKKj=exploitation_table["EBE et revenu mixte brut (1)"]
pKKj=shorten(pKKj)

#distribute Y taxes, transfer and margins across L and K
transfer = exploitation_table["Total des transferts"]
transfer=shorten(transfer)

taxes_production=exploitation_table["Autres impôts sur la production"]+exploitation_table["Autres subv. sur la production"]
taxes_production=shorten(taxes_production)

proportionL=pLLj/(pKKj+pLLj)

pLLj=pLLj+(taxes_production+transfer)*proportionL
pKKj=pKKj+(taxes_production+transfer)*(1-proportionL)



#intermediate consumption

pSiYij=intermediate_cons_table
pSiYij=pSiYij.drop(["TOTAL" ],axis=1)
pSiYij=pSiYij.drop(["PCHTR ", "PCAFAB","TOTAL "],axis=0)
pSiYij.astype(float)


trans_margins= resources_table["Marges de transport"]
trans_margins=shorten(trans_margins)

#pSiYij.loc["GZ"]= pSiYij.loc["GZ"]+com_margins
pSiYij.loc["HZ"]= pSiYij.loc["HZ"]+trans_margins

pSiYij=pSiYij.to_numpy()


#check goods poorly consumed by final consumers 

quartile = np.percentile(employment_table['Ménages'], 25)

employment_table.loc[(employment_table['Ménages'] < quartile) & (employment_table['Ménages']>0)][['Activity','Ménages']]

#check for equilibrium

cons_cost_diff=pSiYij.sum(axis=0)+pLLj+pKKj-(pSiYij.sum(axis=1)+pCjCj+pCjGj+pCjIj-pMjMj+pXjXj)

#list(exploitation_table.index[:-1])
cons_cost_diff=dict(zip(list(exploitation_table.index[:-1]),cons_cost_diff))

 

# for i in intermediate_cons_table.columns:
#     for j in intermediate_cons_table.index:
#         if intermediate_cons_table[i][j]<0:
#             print(i,j)
            
# for i in range(len(pSiYij)):
#     for j in range(len(pSiYij[0])):
#         if pSiYij[i][j]<0:
#             print(i,j)            

#build the indexes for matrix format imaclim-py
sectors=list(resources_table.index)[:-3]
colA=sectors
colZ=[s+'-KL' for s in sectors]
colD=sectors+list(["C","G","I"])
colX=sectors

rowA=sectors
rowZ=[s+'-KL' for s in sectors]
rowD=sectors+list(["K","L"])
rowM=sectors

index1col = ["A"]*len(colA)+["Z"]*len(colZ)+["D"]*len(colD)+["X"]*len(colX)
index2col = colA+colZ+colD+colX
index0col=["N"]*len(index1col)

index1row = ["A"]*len(rowA)+["Z"]*len(rowZ)+["D"]*len(rowD)+["M"]*len(rowM)
index2row = rowA+rowZ+rowD+rowM
index0row=["N"]*len(index1row)

multi_index_matrix=pd.DataFrame(0,index=[index0row, index1row, index2row], columns=[index0col, index1col, index2col])


multi_index_matrix.loc[('N'),('A'), ('D', sectors)] = pSiYij
multi_index_matrix.loc[('N'),('A'), ('D','C')] = pCjCj
multi_index_matrix.loc[('N'),('A'), ('D','G')] = pCjGj
multi_index_matrix.loc[('N'),('A'), ('D','I')] = pCjIj
multi_index_matrix.loc[('N'),('A'), ('X')] = np.diag(pXjXj)
multi_index_matrix.loc[('N'),('D','K'), ('Z')] = pKKj
multi_index_matrix.loc[('N'),('D','L'), ('Z')] = pLLj
multi_index_matrix.loc[('N'),('M'), ('A')] = np.diag(pMjMj)
multi_index_matrix.loc[('N'),('D', sectors),('A')] = np.diag([float("nan")]*len(sectors))
multi_index_matrix.loc[('N'),('Z'),('D', sectors)] = np.diag([float("nan")]*len(sectors))

multi_index_matrix.to_csv("matrix.csv")

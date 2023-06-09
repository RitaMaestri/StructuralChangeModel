import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resources_table=pd.read_csv("data/INSEE FRA/ressources.csv")
resources_table.set_index('Activity code',inplace=True)

employment_table=pd.read_csv("data/INSEE FRA/emplois.csv")
employment_table.set_index('Activity code',inplace=True)

intermediate_cons_table=pd.read_csv("data/INSEE FRA/intermediaire.csv")
intermediate_cons_table.set_index("Branche",inplace=True)

production_table=pd.read_csv("data/INSEE FRA/production.csv")
production_table.set_index('Branche',inplace=True)
production_table=production_table.transpose()

exploitation_table=pd.read_csv("data/INSEE FRA/exploitation.csv")
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

sectors=employment_table.index.values[:-3]
sectors_names=list(employment_table["Activity"][:-3].values)
np.where(sectors=="DZ")[0].item()


####### CONSUMPTION ##########

households=employment_table["Ménages"]
households=clean_data(households)

government=employment_table["Total APU"] + employment_table["ISBLSM"]
government= clean_data(government)

investment = employment_table["FBC totale "] 
investment = clean_data(investment)

export=employment_table["Exportations de biens et de services"]
export = clean_data(export)


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
pCjCj=pCjCj-(com_margins)*(pCjCj/total_domestic_production)
pCjGj=pCjGj-(com_margins)*(pCjGj/total_domestic_production)
pCjIj=pCjIj-(com_margins)*(pCjIj/total_domestic_production)
pXjXj=pXjXj-(com_margins)*(pXjXj/total_domestic_production)


#### PRODUCTION #######

#Labour
pLLj=exploitation_table["Rémunération des salariés"]
pLLj=shorten(pLLj)

#Capital
pKKj=exploitation_table["EBE et revenu mixte brut (1)"]
pKKj=shorten(pKKj)

#CORRECTION: transfer
transfer = exploitation_table["Total des transferts"]
transfer=shorten(transfer)

proportionL=pLLj/(pKKj+pLLj)

pLLj=pLLj+(transfer)*proportionL
pKKj=pKKj+(transfer)*(1-proportionL)


#production taxes
production_taxes=exploitation_table["Autres impôts sur la production"]+exploitation_table["Autres subv. sur la production"]
production_taxes=shorten(production_taxes)

#intermediate consumption
pCiYij=intermediate_cons_table
pCiYij=pCiYij.drop(["TOTAL" ],axis=1)
pCiYij=pCiYij.drop(["PCHTR ", "PCAFAB","TOTAL "],axis=0)
pCiYij.astype(float)

#CORRECTION: trade margins
trans_margins= resources_table["Marges de transport"]
trans_margins=shorten(trans_margins)

#pCiYij.loc["GZ"]= pCiYij.loc["GZ"]+com_margins
pCiYij.loc["HZ"]= pCiYij.loc["HZ"]+trans_margins

pCiYij=pCiYij.to_numpy()


#### DERIVED QUANTITIES #######

#gross domestic product
pYjYj=pCiYij.sum(axis=0)+pLLj+pKKj+production_taxes

#KL good
pKLjKLj=pKKj+pLLj

#Armington good
pSjSj=pCjCj+pCjGj+pCjIj+pCiYij.sum(axis=1)-sales_taxes

#domestic good
pDjDj=pSjSj-pMjMj

#labour taxes
labor_taxes=0

####### CHECK FOR GOODS POORLY CONSUMED BY HOUSEHOLDS ########

quartile = np.percentile(households, 20)

employment_table.loc[(employment_table['Ménages'] < quartile)][['Activity','Ménages']]


######## CHECK FOR EQUILIBRIUM ###########

cons_cost_diff= (
    pCiYij.sum(axis=0)+pLLj+pKKj+production_taxes+sales_taxes-(
    pCiYij.sum(axis=1)+pCjCj+pCjGj+pCjIj-pMjMj+pXjXj
    )
    )

#list(exploitation_table.index[:-1])
#cons_cost_diff=dict(zip(list(exploitation_table.index[:-1]),cons_cost_diff))

total_domestic_production=pCjCj+pCjGj+pCjIj
pCjCj=pCjCj+(cons_cost_diff)*(pCjCj/total_domestic_production)
pCjGj=pCjGj+(cons_cost_diff)*(pCjGj/total_domestic_production)
pCjIj=pCjIj+(cons_cost_diff)*(pCjIj/total_domestic_production)

diff_final= (
    pCiYij.sum(axis=0)+pLLj+pKKj+production_taxes+sales_taxes-(
    pCiYij.sum(axis=1)+pCjCj+pCjGj+pCjIj-pMjMj+pXjXj
    )
    )


pSjSj=pCjCj+pCjGj+pCjIj+pCiYij.sum(axis=1)-sales_taxes


pDjDj=pSjSj-pMjMj

pSjSj-pMjMj - (pYjYj-pXjXj)

non_zero_index_G=np.array(np.where(pCjGj != 0)).flatten()
non_zero_index_I=np.array(np.where(pCjIj != 0)).flatten()
non_zero_index_X=np.array(np.where(pXjXj != 0)).flatten()
non_zero_index_M=np.array(np.where(pMjMj != 0)).flatten()
non_zero_index_C=np.array(np.where(pCjCj != 0)).flatten()
non_zero_index_Yij=np.array(np.where(pCiYij != 0))
len(pCiYij[pCiYij != 0])

N=len(pLLj)

#trade elasticities

export_elasticities=pd.read_csv("data/Export elasticities.csv")

sigmaXj=np.array(export_elasticities["elasticity"])

Armington_elasticities=pd.read_csv("data/Armington_elasticities.csv")

sigmaSj=np.array(Armington_elasticities["elasticity"])

#consumption elasticities

consumption_elasticities=pd.read_csv("data/revenue_elasticities2.csv")

epsilonRj=np.array(consumption_elasticities['Revenue elasticity of consumption'])

epsilonPCj=np.array(consumption_elasticities['Price elasticity of consumption'])


energy_intensity=pd.DataFrame(pCiYij[np.where(sectors=="DZ")[0].item()], index=sectors_names)/sum(pCiYij[np.where(sectors=="DZ")[0].item()])

percentile=np.percentile(energy_intensity, 75)
energy_intensity.loc[energy_intensity[0]>percentile]

# for i in intermediate_cons_table.columns:
#     for j in intermediate_cons_table.index:
#         if intermediate_cons_table[i][j]<0:
#             print(i,j)
            
# for i in range(len(pCiYij)):
#     for j in range(len(pCiYij[0])):
#         if pCiYij[i][j]<0:
#             print(i,j)            














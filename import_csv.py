import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resources_table=pd.read_csv("data/ressources.csv")
resources_table.set_index('Activity code',inplace=True)

employment_table=pd.read_csv("data/emplois.csv")
employment_table.set_index('Activity code',inplace=True)

intermediate_cons_table=pd.read_csv("data/intermediaire.csv")
intermediate_cons_table.set_index('Unnamed: 0',inplace=True)

production_table=pd.read_csv("data/production.csv")
production_table.set_index('Branche',inplace=True)
production_table=production_table.transpose()

exploitation_table=pd.read_csv("data/exploitation.csv")
exploitation_table.set_index('Branche',inplace=True)
exploitation_table=exploitation_table.transpose()

def shorten(db):
    return np.array(db.drop(["PCHTR ", "PCAFAB", "TOTAL", "TOTAL "],errors='ignore'))

def clean_data(db):
    short_db = shorten(db)
    norm_db= np.array([float(i)/sum(short_db) for i in short_db])
    return np.array(short_db) + norm_db * sum(db[["PCHTR ", "PCAFAB"]])


# consumptions
households=employment_table["Ménages"]
households=clean_data(households)

government=employment_table["Total APU"] + employment_table["ISBLSM"]
government= clean_data(government)

investment = employment_table["FBC totale "] 
investment = clean_data(investment)

export=employment_table["Exportations de biens et de services"]
export = clean_data(export)

imports= resources_table["Importations de biens et de services"]
imports=clean_data(imports)

sales_taxes = resources_table["Impôts sur les produits - total -"] + resources_table["Subventions sur les produits"]
sales_taxes=clean_data(sales_taxes)
#production

pCjCj=households+government+investment+export-imports-sales_taxes


pYjYj=resources_table["Production des produits (1)"]
pYjYj=clean_data(pYjYj)

#distribute Y taxes across L and K

pLLj=exploitation_table["Rémunération des salariés"]
pLLj=shorten(pLLj)

pKKj=exploitation_table["EBE et revenu mixte brut (1)"]
pKKj=shorten(pKKj)

taxes_production=exploitation_table["Autres impôts sur la production"]+exploitation_table["Autres subv. sur la production"]
taxes_production=shorten(taxes_production)

proportionL=pLLj/(pKKj+pLLj)

pLLj=pLLj+taxes_production*proportionL
pKKj=pKKj+taxes_production*(1-proportionL)

#other quantities

pKLjKLj=pLLj+pKKj


KLj=pLLj+pKKj


pYj=np.ones(len(pYjYj))


pYiYij=intermediate_cons_table
pYiYij=pYiYij.drop(["TOTAL" ],axis=1)
pYiYij=pYiYij.drop(["PCHTR ", "PCAFAB","TOTAL "],axis=0)
pYiYij.astype(float)
pYiYij=pYiYij.to_numpy()

K=sum(pKKj)
pL=1


plt.hist(employment_table.loc[(employment_table['Ménages'] < 1e5) & (employment_table['Ménages']>0)]['Ménages'])

quartile = np.percentile(employment_table['Ménages'], 25)

employment_table.loc[(employment_table['Ménages'] < quartile) & (employment_table['Ménages']>0)][['Activity','Ménages']]



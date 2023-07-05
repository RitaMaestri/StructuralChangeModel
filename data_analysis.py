#Data Analysis
import simple_calibration as cal
import csv
import pandas as pd
import numpy as np
import import_csv as imp
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick



########## RECREATE THE DICTIONARIES #####################

sectors_names_eng=[
    "AGRICULTURE",
    "EXTRACTIVE IND.",
    "FOOD IND.",
    "TEXTILE IND.",
    "PAPER & WOOD IND.",
    "REFINERIES",
    "CHEMICAL IND.",
    "PHARMACEUTICAL IND.",
    "CAOUTCHOU, PLASTIC IND.",
    "METALLURGIC IND.",
    "INFORMATICS",
    "ELECTRONICS",
    "MACHINES",
    "TRANSPORT MATERIAL",
    "OTHER MANIFACTURING IND.",
    "ELECTRICITY",
    "WATER & WASTE",
    "CONSTRUCTION",
    "TRADE",
    "TRANSPORTATION",
    "HOTELS & RESTAURANTS",
    "PUBLISHING IND.",
    "TELECOMUNICATIONS",
    "INFORMATION SERVICES",
    "FINANCE & INSURANCE",
    "REAL ESTATE",
    "LAWYERS & ENGENEERS",
    "R&D",
    "SCIENTIFIC ACTIVITIES",
    "ADMINISTRATION SERVICES",
    "PUBLIC ADMINISTRATION",
    "TEACHING",
    "HEALTH",
    "SOCIAL SERVICES",
    "RECREATION & ART",
    "OTHER SERVICES",
    "HOUSEHOLD ACTIVITIES"
    ]


df1 = pd.read_csv("results/classic_1.csv").iloc[:,1:]
dfN = pd.read_csv("results/classic_N.csv").iloc[:,1:]
dfN2 = pd.read_csv("results/classic_Yij.csv").iloc[:,1:]

dcl=df1.to_dict(orient="records")[0]

for key in dfN:
    dcl[key]=np.array(dfN[key])
dcl['Yij']=dfN2.to_numpy()    

df1CDES = pd.read_csv("results/CDES_1.csv").iloc[:,1:]
dfNCDES = pd.read_csv("results/CDES_N.csv").iloc[:,1:]
dfN2CDES = pd.read_csv("results/CDES_Yij.csv").iloc[:,1:]


dCDES=df1CDES.to_dict(orient="records")[0]

for key in dfNCDES:
    dCDES[key]=np.array(dfNCDES[key])
    
dCDES['Yij']=dfN2CDES.to_numpy()    


########### COMPARISON TABLES ###############

cons_share_change=(dCDES['alphaCj']-cal.alphaCj)/cal.alphaCj

production_change=(dCDES['Yj']*dCDES['pYj']/(dcl['Yj']*dcl['pYj'])-1)

import_change=(dCDES['Mj']*dCDES['pMj']/(dcl['Mj']*dcl['pMj'])-1)

export_change=(dCDES['Xj']*dCDES['pMj']/(dcl['Xj']*dcl['pMj'])-1)

armington_change=(dCDES['Sj']*dCDES['pSj']/(dcl['Sj']*dcl['pSj'])-1)

consumption_change=(dCDES['Cj']*dCDES['pCj']/(dcl['Cj']*dcl['pCj'])-1)

consumption_quantity_change=(dCDES['Cj']/(dcl['Cj'])-1)

consumption_price_change=(dCDES['pCj']/(dcl['pCj'])-1)

production_price_change=(dCDES['pYj']/(dcl['pYj'])-1)


df=pd.DataFrame(data=np.transpose(
    [imp.sectors,
     production_change,
     import_change, 
     export_change,
     armington_change,
     consumption_change,
     consumption_quantity_change,
     consumption_price_change,
     cal.epsilonRj,
     cal.epsilonPCj,
     cal.alphaCj,
     ((dCDES['alphaCj']-cal.alphaCj)/cal.alphaCj) ,
     dcl['Yij'].sum(axis=1)
     ]), 
    index=imp.sectors_names, 
    columns=["code","production","import","export","armington","consumption","consumption_quantity_change","price","epsilonR", "cal.epsilonPCj","alphaCj", "change alphaCj", "sumYij"]
    ).fillna(0)


#### difference between the two specifications ####

# KL GDP decomposition 

GDP0decomposition = imp.pLLj+imp.pKKj
GDP1decomposition = dcl['pL']*dcl['Lj']+dcl['pK']*dcl['Kj']
decomposition_difference=(GDP1decomposition/sum(GDP1decomposition)-GDP0decomposition/sum(GDP0decomposition))/(GDP0decomposition/sum(GDP0decomposition))

# consumption GDP decomposition 
GDP0decomposition = imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj
GDP1decomposition = dcl['pCj']*(dcl['Cj']+dcl['Gj']+dcl['Ij'])+dcl['pMj']*dcl['Xj']-dcl['pMj']*dcl['Mj']
decomposition_difference=(GDP1decomposition/sum(GDP1decomposition)-GDP0decomposition/sum(GDP0decomposition))/(GDP0decomposition/sum(GDP0decomposition))


######### PLOTS ##########

##### intermediate sales ####### (row)
index=imp.pSiYij.sum(axis=1).argsort()
name = imp.sectors[index]

index=GDP0decomposition.argsort()
name = imp.sectors[index]

plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(1, figsize=(16, 10))

plt.barh(name, imp.pSiYij.sum(axis=1)[index])

ax.set_title('Intermediate sales by sector',
             loc ='center', )

plt.show()

##### intermediate purchases ####### (column)
index=imp.pSiYij.sum(axis=0).argsort()

plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(1, figsize=(16, 10))

plt.barh(name, imp.pSiYij.sum(axis=0)[index])


ax.set_title('Intermediate purchase by sector',
             loc ='center', )

plt.show()


### GDP KL decomposition #######
index=GDP0decomposition.argsort()
name = imp.sectors[index]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, figsize=(16, 10))

plt.barh(name, imp.pLLj[index])
plt.barh(name, imp.pKKj[index], left= imp.pLLj[index])

plt.ylim(0,len(index))
ax.set_title('GDP sectoral decomposition in factors of production',
             loc ='center', )

plt.legend(['Labour','Capital'], loc='lower right', ncol = 5)

plt.show()

### pYY #######

index=imp.pYjYj.argsort()
name = imp.sectors[index]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, figsize=(16, 10))

plt.barh(name, imp.pYjYj[index])


ax.set_title('Gross domestic output ($p_Y Y$)',
             loc ='center', )

plt.show()

# pYY decomposed into KL and intermediate purchases
plt.barh(name,  imp.pKLjKLj[index])
plt.barh(name, imp.pSiYij.sum(axis=0)[index], left= imp.pKLjKLj[index])


ax.set_title('Gross domestic output ($p_Y Y$)',
             loc ='center', )

plt.legend(['KL', 'Intermediate Purchases'], loc='lower right', ncol = 5)


plt.show()



#### Cobb douglas GDP decomposition
GDP0decomposition = imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj

index=GDP0decomposition.argsort()
name = imp.sectors[index]

fig, ax = plt.subplots(1, figsize=(16, 10))

plt.barh(name, dcl['pCj']*dcl['Cj'][index])
plt.barh(name, dcl['pCj']*dcl['Gj'][index], left= dcl['pCj']*dcl['Cj'][index])
plt.barh(name, dcl['pCj']*dcl['Ij'][index],left= dcl['pCj']*dcl['Cj'][index]+dcl['pCj']*dcl['Gj'][index] )
plt.barh(name, dcl['pMj']*dcl['Xj'][index], left = dcl['pCj']*dcl['Cj'][index]+dcl['pCj']*dcl['Gj'][index]+dcl['pCj']*dcl['Ij'][index])
plt.barh(name,-dcl['pMj']*dcl['Mj'][index])

ax.set_title('Sectoral GDP decomposition with Cobb Douglas',
             loc ='center', )

plt.legend(['Households', 'Government', 'Investment', 'Export', 'Import'], loc='lower right', ncol = 5)

plt.show()



###Comparison of C+G+I+X-M (quantities) for baseline year, CD and CDES ###
GDP0decomposition = imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj
order=GDP0decomposition.argsort()

n_groups = len(imp.sectors)
consumption_baseline = (cal.Cj0+cal.Gj0+cal.Ij0+cal.Xj0-cal.Mj0)[order]
consumption_CDES = (dCDES['Cj']+dCDES['Gj']+dCDES['Ij']+dCDES['Xj']-dCDES['Mj'])[order]
consumption_Cobb = (dcl['Cj']+dcl['Gj']+dcl['Ij']+dcl['Xj']-dcl['Mj'])[order]


fig, ax =  plt.subplots(1, figsize=(15, 20))
index = np.arange(n_groups)
bar_width=0.22

plt.rcParams.update({'font.size': 20})


bar1 = plt.barh(index, consumption_baseline,bar_width, label='Calibration year')

bar2 = plt.barh(index + bar_width, consumption_Cobb,bar_width, label='Cobb Douglas')

bar3= plt.barh(index + 2*bar_width, consumption_CDES,bar_width, label='CDES')

plt.ylim(0,len(index))
plt.title('Quantity of domestic product consumption by sector (C+G+I+X-M)')
plt.yticks(index + 1.5*bar_width, imp.sectors[order])


plt.tight_layout()

plt.legend(loc='lower right')

plt.show()

###Comparison of KL (quantities) for baseline year, CD and CDES ###


order=cal.KLj0.argsort()


n_groups = len(imp.sectors)
KL_baseline = cal.KLj0[order]
KL_CDES = dCDES['KLj'][order]
KL_Cobb = dcl['KLj'][order]


fig, ax =  plt.subplots(1, figsize=(15, 20))
index = np.arange(n_groups)
bar_width=0.22

plt.rcParams.update({'font.size': 20})


bar1 = plt.barh(index, KL_baseline,bar_width, label='Calibration year')

bar2 = plt.barh(index + bar_width, KL_Cobb,bar_width, label='Cobb Douglas')

bar3= plt.barh(index + 2*bar_width, KL_CDES,bar_width, label='CDES')

plt.ylim(0,len(index))
plt.title('Capital-Labour quantity by sector (KL) ')
plt.yticks(index + 1.5*bar_width, imp.sectors[order])


plt.tight_layout()

plt.legend(loc='lower right')

plt.show()


###Comparison of Y (quantities) for baseline year, CD and CDES ###
order=cal.Yj0.argsort()

n_groups = len(imp.sectors)
Y_baseline = cal.Yj0[order]
Y_CDES = dCDES['Yj'][order]
Y_Cobb = dcl['Yj'][order]


fig, ax =  plt.subplots(1, figsize=(15, 20))
index = np.arange(n_groups)
bar_width=0.22

plt.rcParams.update({'font.size': 20})


bar1 = plt.barh(index, Y_baseline,bar_width, label='Calibration year')

bar2 = plt.barh(index + bar_width, Y_Cobb,bar_width, label='Cobb Douglas')

bar3= plt.barh(index + 2*bar_width, Y_CDES,bar_width, label='CDES')

plt.ylim(0,len(index))
plt.title('Gross domestic output quantity by sector (Y)')
plt.yticks(index + 1.5*bar_width, imp.sectors[order])


plt.tight_layout()

plt.legend(loc='lower right')

plt.show()


###Comparison of Yij (quantities) for baseline year, CD and CDES ###

order = cal.Yij0.sum(axis=1).argsort()

n_groups = len(imp.sectors)
Yij_baseline = cal.Yij0.sum(axis=1)[order]
Yij_CDES = dCDES['Yij'].sum(axis=1)[order]
Yij_Cobb = dcl['Yij'].sum(axis=1)[order]


fig, ax =  plt.subplots(1, figsize=(15, 20))
index = np.arange(n_groups)
bar_width=0.22

plt.rcParams.update({'font.size': 20})


bar1 = plt.barh(index, Yij_baseline,bar_width, label='Calibration year')

bar2 = plt.barh(index + bar_width, Yij_Cobb,bar_width, label='Cobb Douglas')

bar3= plt.barh(index + 2*bar_width, Yij_CDES,bar_width, label='CDES')

plt.ylim(0,len(index))
plt.title('Intermediate sales quantity by sector')
plt.yticks(index + 1.5*bar_width, imp.sectors[order])


plt.tight_layout()

plt.legend(loc='lower right')

plt.show()



### Composition of Y (KL or intermediate demand)###

dfKLY=pd.DataFrame({"sectors":imp.sectors[order],"Capital-Labour contribution":cal.aKLj[order],"Intermediate supply contribution": cal.aYij.sum(axis=0)[order]})
plt.rcParams.update({'font.size':20})

dfKLY.plot(
    x = 'sectors',
    
    kind = 'barh',
    stacked = True,
    title = 'Composition of gross domestic output Y',
    mark_right = True,
    figsize=(15,15)
    )
plt.legend(loc='lower right')

####### PERCENTAGE VARIATIONS CDES ########


##### KL percentage variation#######

GDP0decomposition = cal.KLj0
#order=(-GDP0decomposition).argsort()
#order=((dCDES['Cj'] - cal.Cj0)/cal.Cj0).argsort()
order=list(range(len(imp.sectors)))

n_groups = len(imp.sectors)
consumption_CDES = (dCDES['KLj'][order] - cal.KLj0[order])/cal.KLj0[order]*100
consumption_Cobb = (dcl['KLj'][order] - cal.KLj0[order])/cal.KLj0[order]*100

fig, ax =  plt.subplots(1, figsize=(25, 15))
index = np.arange(n_groups)
bar_width=0.9

plt.rcParams.update({'font.size': 25})

bar1 = plt.bar(index , consumption_CDES, bar_width, color='green')
#bar2= plt.barh(index + bar_width, consumption_Cobb, bar_width, label='Cobb Douglas')

plt.title('Percentage variation of the value-added (KL) quantity with respect to the calibration year')
plt.xlim(-0.5,len(index)-0.5)

plt.xticks(index+0.4, np.array(sectors_names_eng)[order],rotation=50,ha="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.axhline(y=10)
# np.zeros(len(imp.sectors))
# for i, (name, height) in enumerate(zip(sectors_names_eng, np.array([1]*len(imp.sectors) ))):
#     ax.text(i, height, ' ' + name, color='seashell',
#             ha='center', va='bottom', rotation=90, fontsize=18, weight='bold')

plt.tight_layout()

plt.show()

##### Y percentage variation #######
plt.rcParams.update({'font.size': 25})
GDP0decomposition = cal.Yj0
#order=(-GDP0decomposition).argsort()
#order=((dCDES['Cj'] - cal.Cj0)/cal.Cj0).argsort()
order=list(range(len(imp.sectors)))

n_groups = len(imp.sectors)
consumption_CDES = (dCDES['Yj'][order] - cal.Yj0[order])/cal.Yj0[order]*100

fig, ax =  plt.subplots(1, figsize=(25, 15))
index = np.arange(n_groups)
bar_width=0.5

bar1 = plt.bar(index , consumption_CDES, bar_width, color='orangered')

plt.title('Percentage variation of the production quantity (Y) with respect to the calibration year')
plt.xlim(-0.5,len(index)-0.5)

plt.xticks(index+0.4, np.array(sectors_names_eng)[order],rotation=50,ha="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.axhline(y=10)

plt.tight_layout()

plt.show()


##### Y percentage variation#######
plt.rcParams.update({'font.size': 25})
#order = (-cal.Yij0.sum(axis=1)).argsort()
#order=((dCDES['Cj'] - cal.Cj0)/cal.Cj0).argsort()
order=list(range(len(imp.sectors)))

n_groups = len(imp.sectors)
consumption_CDES = (dCDES['Yij'].sum(axis=1)[order] - cal.Yij0.sum(axis=1)[order])/cal.Yij0.sum(axis=1)[order]*100

fig, ax =  plt.subplots(1, figsize=(25, 15))
index = np.arange(n_groups)
bar_width=0.5

bar1 = plt.bar(index , consumption_CDES, bar_width, color='navy')

plt.title('Percentage variation of the intermediate sale quantity ($\sum_j Y_{ij}$) with respect to the calibration year')
plt.xlim(-0.5,len(index)-0.5)

plt.xticks(index+0.4, np.array(sectors_names_eng)[order],rotation=50,ha="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.axhline(y=10)

plt.tight_layout()

plt.show()


##### C percentage variation#######
plt.rcParams.update({'font.size': 25})

order=list(range(len(imp.sectors)))


n_groups = len(imp.sectors)
consumption_CDES = (dCDES['Cj'][order] - cal.Cj0[order])/cal.Cj0[order]*100

fig, ax =  plt.subplots(1, figsize=(25, 15))
index = np.arange(n_groups)
bar_width=0.5

bar1 = plt.bar(index , consumption_CDES, bar_width, color='orange')


plt.title('Percentage variation of the household consumption quantity (C) with respect to the calibration year')
plt.xlim(-0.5,len(index)-0.5)

plt.xticks(index+0.4, imp.sectors[order],rotation=50,ha="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.axhline(y=10)

plt.tight_layout()


plt.show()

a=(dCDES['Yj'][order] - cal.Yj0[order])/cal.Yj0[order]*100-(dCDES['Cj'][order] - cal.Cj0[order])/cal.Cj0[order]*100
a.argsort()
sectors_names_eng[19]


######## COMPARISON TABLES #################

dfM=pd.DataFrame(data=np.transpose(
    [cal.sigmaSj,
     imp.pLLj/(imp.pLLj+imp.pKKj),
     cal.aKLj,
     dcl['pDj'],
     dcl['pYj'], 
     dcl['pMj'], 
    (dcl['pDj']*dcl['Dj']/sum(dcl['pDj']*dcl['Dj']) - imp.pYjYj / sum(imp.pDjDj) )/(imp.pDjDj / sum(imp.pDjDj))*100,
     
    (dcl['pYj']*dcl['Yj']/sum(dcl['pYj']*dcl['Yj']) - imp.pYjYj / sum(imp.pYjYj) )/(imp.pYjYj / sum(imp.pYjYj))*100,
     (dcl['pMj']*dcl['Mj']/sum(dcl['pMj']*dcl['Mj']) - imp.pMjMj / sum(imp.pMjMj) )/(imp.pMjMj / sum(imp.pMjMj))*100,
     (dcl['pMj']*dcl['Xj']/sum(dcl['pMj']*dcl['Xj'])- imp.pXjXj / sum(imp.pXjXj) )/(imp.pMjMj / sum(imp.pMjMj))*100
     ]),
    index=imp.sectors_names, 
    columns=["sigmaSj","pL/pKL","aKL","pDj","pYj","pM","pDD","Y","change pMM","pXX"]
    ).fillna(0)


# 100*((imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)/cal.GDP0-(dcl['pCj']*(dcl['Cj']+dcl['Gj']+dcl['Ij'])+dcl['pMj']*dcl['Xj']-dcl['pMj']*dcl['Mj'])/cal.GDPreal)/((imp.pCjCj+imp.pCjGj+imp.pCjIj+imp.pXjXj-imp.pMjMj)/cal.GDP0)
# dcl['Yj']*dcl['pYj']/sum(dcl['Yj']*dcl['pYj'])-imp.pYjYj/sum(imp.pYjYj)


dfprices=pd.DataFrame(data=
    [(dcl['pL']-cal.pL0)/cal.pL0,
     (dcl['pK']-cal.pK0)/cal.pK0,
     np.mean(abs((dcl['pKLj']-cal.pKLj0))/cal.pKLj0),
     np.mean(abs((dcl['pYj']-cal.pYj0))/cal.pYj0),
     np.mean(abs((dcl['pDj']-cal.pDj0))/cal.pDj0),
     np.mean(abs((dcl['pSj']-cal.pDj0))/cal.pDj0),
      ],
    index=["pL","pK","pKL","pY","pD","pS"], 
        )


dfquantities=pd.DataFrame(data=
    [np.mean(abs((dcl['Lj']-cal.Lj0))/cal.Lj0),  
     np.mean(abs((dcl['Kj']-cal.Kj0))/cal.Kj0), 
     np.mean(abs((dcl['Xj']-cal.Xj0))/cal.Xj0),
     np.mean(abs((dcl['Mj']-cal.Mj0))/cal.Mj0),
      ],
    index=["pL","pK","pKL","pY","pD","pS"], 
        )






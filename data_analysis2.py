#DATA ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import import_csv as imp
import sys
from simple_calibration import N

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

name = imp.sectors

df = pd.read_csv("results/neoclassic2015-2050.csv")
df.rename(columns={df.columns[0]: 'variable'},inplace=True)



diff= False
var='KLj'
#"p","q","pq"
pq="q"





P=df.loc[ df['variable'] == "p"+var ].reset_index(drop=True).drop("variable", axis=1)
Q=df.loc[ df['variable'] == var ].reset_index(drop=True).drop("variable", axis=1)


if pq=="pq":
    VA = P*Q
elif pq == "p":
    VA = P
elif pq == "q":
    VA= Q
else:
    print("wrong pq!")
    sys.exit()


# start_range=sectors_names_eng.index("PHARMACEUTICAL IND.")
# this_range=range(N*start_range,N*start_range+N)

# var="Yij"

# Q=df.loc[ df['variable'] == "Yij" ].reset_index(drop=True).drop("variable", axis=1).loc[this_range]

# VA=Q

VA.insert(0, "Name", sectors_names_eng, True)

for i in np.split(VA.index, [9,18,27]):
    for j in i:
        x=np.array(VA.columns[2:]).astype('int')
        y=np.array(VA.loc[j])[2:].astype('float')
        y= y-shift(y, 1, cval=np.NaN) if diff else y
        plt.plot(x[1:],y[1:] , label = VA.loc[j][0])
    ax = plt.subplot(111)
    
    # Shrink current axis's height by 10% on the bottom
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if pq=="pq":
        plt.title("p"+var+var)
    elif pq == "p":
        plt.title("p"+var)
    elif pq == "q":
        plt.title(var)
    

    plt.show()



pharma_ind=sectors_names_eng.index("PHARMACEUTICAL IND.")
agri_ind=sectors_names_eng.index("AGRICULTURE")
    
#par_keys = ['aKLj', 'alphaCj', 'alphaDj','alphaGj','alphaIj', 'alphaKj', 
#            'alphaLj','alphaXj','betaDj','betaMj', 'bKLj', 'tauYj', 'tauSj']

par_keys = ['tauSj', 'tauYj', 'pXj', 'bKLj', 'alphaKj', 'alphaLj', 'aKLj', 'alphaCj', 
'alphaGj', 'alphaIj', 'alphaXj', 'alphaDj', 'betaDj', 'betaMj', 
'sigmaXj', 'sigmaSj']


for i in par_keys:
    array=np.array(df.loc[(df['variable'] == i)].iloc[:, 1])
    order = array.argsort()
    print(i,36-18-np.where(order==pharma_ind)[0].item(),"\n")


#plot capital and labour prices
pL=df.loc[ df['variable'] == "pL" ].reset_index(drop=True).drop("variable", axis=1)
pK=df.loc[ df['variable'] == "pK" ].reset_index(drop=True).drop("variable", axis=1)

x=np.array(VA.columns[1:]).astype('int')
y=np.array(pK)[0].astype('float')
y= y-shift(y, 1, cval=np.NaN) if diff else y
plt.plot(x[1:],y[1:] , label = "pL")
plt.title("pK")

#plot GDP growth, capital and labour growth
L=np.array(df.loc[(df['variable'] == "L")].iloc[:,1:])[0].astype("float")
K=np.array(df.loc[(df['variable'] == "K")].iloc[:,1:])[0].astype("float")
GDPreal=np.array(df.loc[(df['variable'] == "GDPreal")].iloc[:,1:])[0].astype("float")

x=np.array(df.columns[1:]).astype('int')
y= GDPreal
plt.plot(x,y,label = "GDPreal")
plt.title("GDPreal")
plt.show()

x=np.array(df.columns[1:]).astype('int')
y= L
plt.plot(x,y,label = "L")
y= K
plt.plot(x,y,label = "K")

ax = plt.subplot(111)

# Shrink current axis's height by 10% on the bottom
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("L & K")
plt.show()







#DATA ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import import_csv as imp
import sys
from simple_calibration import N
import matplotlib.colors as mcolors

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

df = pd.read_csv("results/johansen2015-2050exoKnext(02-03-2023_18:37).csv", index_col=0)
df.rename(columns={df.columns[0]: 'variable'},inplace=True)

df.loc[ df['variable'] == "L" ].drop("variable", axis=1)
a=df.loc[ df['variable'] == "bKL" ].drop("variable", axis=1)


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

VA.insert(0, "Name", sectors_names_eng, True)

# start_range=sectors_names_eng.index("PHARMACEUTICAL IND.")
# this_range=range(N*start_range,N*start_range+N)

# var="Yij"

# Q=df.loc[ df['variable'] == "Yij" ].reset_index(drop=True).drop("variable", axis=1).loc[this_range]

# VA=Q

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)

x=np.array(VA.columns[2:]).astype('int')

y0= x[1].astype(str)
yL=x[-1].astype(str)

rank= np.argsort(VA[y0].values/VA[yL].values)
index_side_writing=np.hstack([rank[:3],rank[-3:]])

for j in VA.index:
    y=np.array(VA.loc[j])[2:].astype('float')
    y= y/y[0]-shift(y/y[0], 1, cval=np.NaN) if diff else y/y[0]
    lab=VA.loc[j][0]
    ax.plot(x[0:],y[0:] , label = lab)
    if j in index_side_writing:
        print(j)
        print(VA.loc[j][0])
        ax.annotate(xy=(x[-1],y[-1]), xytext=(5,0), textcoords='offset points', text=VA.loc[j][0], va='center')

    
    
colors1=plt.cm.Accent(np.linspace(0., 1, 128))
colors2=plt.cm.tab10(np.linspace(0., 1, 128))
colors3=plt.cm.Set3(np.linspace(0.1, 1, 128))
colors4=plt.cm.gist_ncar(np.linspace(0., .8, 128))
mycolors = np.vstack((colors1, colors2, colors3, colors4))
colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', mycolors)


 #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
for i,j in enumerate(ax.lines):
    j.set_color(colors[i])
    
# Shrink current axis's height by 10% on the bottom
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# # Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1.16, 0.5), prop={'size': 16})

if pq=="pq":
    plt.title("p"+var+var)
elif pq == "p":
    plt.title("p"+var)
elif pq == "q":
    plt.title(var)
plt.xlim(2014.99,2050.01)

plt.show()





for i in np.split(VA.index, [9,18,27]):
    for j in i:
        x=np.array(VA.columns[1:]).astype('int')
        y=np.array(VA.loc[j])[1:].astype('float')
        y= y/y[0]-shift(y/y[0], 1, cval=np.NaN) if diff else y/y[0]
        plt.plot(x[0:],y[0:] , label = VA.loc[j][0])
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
plt.show()



#plot GDP growth, capital and labour growth
L=np.array(df.loc[(df['variable'] == "L")].iloc[:,1:])[0].astype("float")
bKL=np.array(df.loc[(df['variable'] == "bKL")].iloc[:,1:])[0].astype("float")
LbKL=L*bKL
K=np.array(df.loc[(df['variable'] == "K")].iloc[:,1:])[0].astype("float")
GDPreal=np.array(df.loc[(df['variable'] == "GDPreal")].iloc[:,1:])[0].astype("float")

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
x=np.array(df.columns[1:]).astype('int')
y= LbKL/LbKL[0]-shift(LbKL/LbKL[0], 1, cval=np.NaN) if diff else LbKL/LbKL[0]
plt.plot(x,y,label = "$L$ x $b_{KL}$")
y= GDPreal/GDPreal[0]-shift(GDPreal/GDPreal[0], 1, cval=np.NaN) if diff else GDPreal/GDPreal[0]
plt.plot(x,y,label = "$GDPreal$")
y= K/K[0]-shift(K/K[0], 1, cval=np.NaN) if diff else K/K[0]
plt.plot(x,y,label = "$K$")



# Shrink current axis's height by 10% on the bottom
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.show()





# x=np.array(df.columns[1:]).astype('int')
# y= L
# plt.plot(x,y,label = "L")
# y= K
# plt.plot(x,y,label = "K")

# ax = plt.subplot(111)

# # Shrink current axis's height by 10% on the bottom
# # Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.title("L & K")
# plt.show()







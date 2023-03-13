#DATA ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
import import_csv as imp
import sys
from simple_calibration import N
import matplotlib.colors as mcolors
from scipy import stats

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


cmap=["#E06969",
"#E80000",
"#B30000",
"#FFA64D",
"#F27900",
"#B35900",
"#FFFF73",
"#BFBF00",
"#7A7A00",
"#B9FF73",
"#6CD900",
"#3D7A00",
"#54F0A2",
"#00BF60",
"#00572B",
"#8FFFFF",
"#19C6FF",
"#007D7D",
"#73B9FF",
"#0060BF",
"#003469",
"#8C8CFF",
"#5959FF",
"#0000FF",
"#000069",
"#B973FF",
"#6300C7",
"#340069",
"#FF00FF",
"#B00CB0",
"#5C005C",
"#9C004E",
"#FF1F8F",
"#FF91C8",
"#000000",
"#545454",
"#9294A1"]

my_cmap=mcolors.ListedColormap(cmap)

name = imp.sectors

df = pd.read_csv("results/johansen2015-2050exoKnext(02-03-2023_18:37).csv")
df.rename(columns={df.columns[0]: 'variable'},inplace=True)

df.loc[ df['variable'] == "L" ].drop("variable", axis=1)
a=df.loc[ df['variable'] == "bKL" ].drop("variable", axis=1)




#pq = "p","q","pq"  
def extract_var_df(var, pq):
    P=df.loc[ df['variable'] == "p"+var ].reset_index(drop=True).drop("variable", axis=1)
    Q=df.loc[ df['variable'] == var ].reset_index(drop=True).drop("variable", axis=1)

    if pq=="pq":
        var_df = P*Q
    elif pq == "p":
        var_df = P
    elif pq == "q":
        var_df= Q
    else:
        print("wrong pq!")
        sys.exit()
    #print(var_df)
    if len(var_df.index)>1:
        var_df.insert(0, "Name", sectors_names_eng, True) 
    return var_df 
    
    


def plot_varj_evol(var, pq, diff=False):
    
    var_df=extract_var_df(var, pq)
    
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111)
    
    x=np.array(var_df.columns[1:]).astype('int')
    
    y0= x[0].astype(str)
    yL= x[-1].astype(str)
    
    
    relative_values=var_df[yL].values/var_df[y0].values
    nans=np.argwhere(np.isnan(relative_values)).flatten()
    #nansX=np.argwhere(np.array(df["2050"].loc[ df['variable'] == "Xj" ]==0)).flatten()
    rank= np.argsort(-relative_values)
    rank=rank[~np.in1d(rank,nans)]
    index_side_writing=np.hstack([rank[:3],rank[-3:]])
    
    
    for j in rank:
        y=np.array(var_df.loc[j])[1:].astype('float')
        y= y/y[0]-shift(y/y[0], 1, cval=np.NaN) if diff else y/y[0]
        lab=var_df.loc[j][0]
        ax.plot(x[0:],y[0:] , label = lab)
        if j in index_side_writing:
            ax.annotate(xy=(x[-1],y[-1]), xytext=(5,0), textcoords='offset points', text=var_df.loc[j][0], va='center')
    
        
        
    # colors1=plt.cm.Accent(np.linspace(0., 1, 128))
    # colors2=plt.cm.tab10(np.linspace(0., 1, 128))
    # colors3=plt.cm.Set3(np.linspace(0.1, 1, 128))
    # colors4=plt.cm.gist_ncar(np.linspace(0., .8, 128))
    # mycolors = np.vstack((colors1, colors2, colors3, colors4))
    # colormap = mcolors.LinearSegmentedColormap.from_list('my_colormap', mycolors)
    
    
    #  #nipy_spectral, Set1,Paired   
    # colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
    
    for i,j in enumerate(ax.lines):
        j.set_color(my_cmap.colors[i])
        
    
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
    
    plt.title("Value added by sector", fontsize = 25)
    plt.xlabel("Year",fontsize = 15)
    plt.ylabel("Relative change with respect to year 2015", fontsize = 15)

    
    plt.show()

plot_varj_evol(var="KLj", pq="q", diff=False)


def plot_splitted_evolutions(var, pq, diff):
    
    var_df=extract_var_df(var, pq)
    for i in np.split(var_df.index, [9,18,27]):
        for j in i:
            x=np.array(var_df.columns[1:]).astype('int')
            y=np.array(var_df.loc[j])[1:].astype('float')
            y= y/y[0]-shift(y/y[0], 1, cval=np.NaN) if diff else y/y[0]
            plt.plot(x[0:],y[0:] , label = var_df.loc[j][0])
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
    

par_keys = ['tauSj', 'tauYj', 'pXj', 'bKLj', 'alphaKj', 'alphaLj', 'aKLj', 'alphaCj', 
'alphaGj', 'alphaIj', 'alphaXj', 'alphaDj', 'betaDj', 'betaMj', 
'sigmaXj', 'sigmaSj']


for i in par_keys:
    array=np.array(df.loc[(df['variable'] == i)].iloc[:, 1])
    order = array.argsort()
    print(i,36-18-np.where(order==pharma_ind)[0].item(),"\n")


################# plot capital and labour prices ###############################




def plot_variable_1D(var_df, title, diff=False):
    fig = plt.figure(figsize=(15,10))
    x=np.array(var_df.columns).astype('int')
    y=np.array(var_df)[0].astype('float')
    y= y-shift(y, 1, cval=np.NaN) if diff else y
    plt.plot(x[1:],y[1:] , label = title)
    plt.title(title, fontsize = 35)
    plt.xlabel("Year",fontsize = 20)
    plt.ylabel("Relative change with respect to year 2015", fontsize = 20)

    plt.show()
    
variable= extract_var_df("GDPreal", "q")
plot_variable_1D(variable, "Real GDP", diff=False)
###############################################################################





################# plot GDP growth, capital and labour growth #######################
L=np.array(df.loc[(df['variable'] == "L")].iloc[:,1:])[0].astype("float")
bKL=np.array(df.loc[(df['variable'] == "bKL")].iloc[:,1:])[0].astype("float")
LbKL=L*bKL
K=np.array(df.loc[(df['variable'] == "K")].iloc[:,1:])[0].astype("float")
GDPreal=np.array(df.loc[(df['variable'] == "GDPreal")].iloc[:,1:])[0].astype("float")

fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)
x=np.array(df.columns[1:]).astype('int')
y=  GDPreal/GDPreal[0]
plt.plot(x,y,label = "$GDP$")
y=  LbKL/LbKL[0]
plt.plot(x,y,label = "$Labour$")

y=  K/K[0]
plt.plot(x,y,label = "$Capital$")

# colors= ["#06646E","#0DABBD","#56C9D6"]
# for i,j in enumerate(ax.lines):
#     j.set_color(colors[i])
# Shrink current axis's height by 10% on the bottom
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
plt.title("Evolution of the parameters from the coupled IAM", fontsize = 25)
plt.xlabel("Year",fontsize = 15)
plt.ylabel("Relative change with respect to year 2015", fontsize = 15)

plt.show()
###############################################################################




######################### numerical analysis of KLj2050 extremes ###############
def order_variablesj (var, relative=False):
    array2015=np.array(df["2016"].loc[ df['variable'] == var ]) 
    array2050=np.array(df["2050"].loc[ df['variable'] == var ])
    array= array2050/array2015 if relative else array2050  
    return(pd.Series(array, index =sectors_names_eng).sort_values())

def value(var, sec="PHARMACEUTICAL IND."):
    array2050=np.array(df["2050"].loc[ df['variable'] == var ])
    series=pd.Series(array2050, index =sectors_names_eng).sort_values()
    return series[sec]

print(value("KLj", "MACHINES"))

par_keys = ['tauSj', 'tauYj', 'pXj', 'bKLj', 'alphaKj', 'alphaLj', 'aKLj', 'alphaCj', 
'alphaGj', 'alphaIj', 'alphaXj', 'alphaDj', 'betaDj', 'betaMj', 
'sigmaXj', 'sigmaSj']

nansX=np.argwhere(np.array(df["2050"].loc[ df['variable'] == "Xj" ]==0)).flatten()


plot_varj_evol(var="KLj", pq="q", diff=False)

order_variablesj("alphaXj")#.drop([sectors_names_eng[i] for i in nans])


def function_pD(pY,sigmaY,alphaX,pX,alphaD):
    Ysigma=1-sigmaY
    pD=(((pY**Ysigma)-(alphaX**sigmaY)*(pX**Ysigma)) / (alphaD**sigmaY))**(1/Ysigma)
    return pD

value("pDj", "AGRICULTURE")

print(function_pD(pY=value("pYj"),
                  sigmaY=value("sigmaXj"),
                  alphaX=value("alphaXj","AGRICULTURE"),
                  pX=value("pXj"),
                  alphaD=value("alphaDj")))


plot_varj_evol(var="KLj", pq="q", diff=False)


###########




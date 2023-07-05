import numpy as np
import pandas as pd
from data_closures import N, calibrationDict
from deepdiff import DeepDiff


def growth_ratio_to_rate(ratio): #the first element of ratio is 1 (at calibration year)
    growth_rates=np.array([np.nan]*(len(ratio)-1))
    growth_rates[0] = ratio[1]-1
    for t in range(1,len(ratio)-1):
        growth_rates[t]= ratio[t+1] /np.prod(1+growth_rates[:t])-1
    return growth_rates
#to be done: define growth_rate_to_ratio

# BUILD DATAFRAME

class sys_df:
    #it creates a dicitonary of parameters' temporal series based on growth rates wrt year 0 or growth ratios
    def evolve_par(self, growth_rate = False):
        
        growth_dictionary = {key: None for key in self.growth_ratios.keys()}
        
        step=self.years[1]-self.years[0]
        
        if growth_rate:
            for i in self.growth_rates.keys():
                par0=[self.calib_par_dict[i]]
                [par0.append( par0[-1]*( 1 + self.growth_rates[i][j])**step ) for j in range(len(self.growth_rates[i])) ]
                par0=np.array(par0)
                growth_dictionary[i]=par0
        else:
            for i in self.growth_ratios.keys():
                if self.growth_ratios[i].ndim == 1 :

                    growth_dictionary[i]=self.growth_ratios[i]*self.calib_par_dict[i]
                else: 
                    repeated_array = np.tile(self.calib_par_dict[i], (np.shape(self.growth_ratios[i])[1], 1)).T
                    growth_dictionary[i] = self.growth_ratios[i]*repeated_array
                    
        return growth_dictionary

    ####  X axis  ####
    
    # define closure : "johansen" , "neoclassic", "kaldorian", "keynes-marshall", "keynes", "keynes-kaldor","neokeynesian1", "neokeynesian2"   ########

    # fill in dataframe at time t
    def empty_dataframe(self,dictionary):
        # build X index
        length_dict = {key: np.prod(np.shape(value)) for key, value in dictionary.items()}
        
        Xindex=list([])
        [ Xindex.extend([key]*int(value)) for key, value in length_dict.items()]
        
        #build empty dataframe
        df = pd.DataFrame(data=None, index=Xindex, columns=self.years)
        
        return df

    def __parameters_dynamics(self):
        static_par = list(set(self.calib_par_dict.keys()) - set(self.dynamic_parameters.keys()))
        dynamic_par = list(self.calib_par_dict.keys() & self.dynamic_parameters.keys())
        
        for key, value in {k: self.calib_par_dict[k] for k in static_par}.items():
                self.parameters_df.loc[key] = np.repeat([value.flatten() if isinstance(value, np.ndarray) else value], int(len(self.years)), axis=0).T
    
        for key in dynamic_par:
                self.parameters_df.loc[key] = self.dynamic_parameters[key]

    def __initialize_variables_df(self):

        self.variables_df= self.empty_dataframe(self.calib_var_dict)
        
        #self.dict_to_df(self.calib_var_dict, self.years[0])

    def __initialize_parameters_df(self):
        
        self.parameters_df=self.empty_dataframe(self.calib_par_dict)
        
        self.__parameters_dynamics()


    def dict_to_df(self,dictionary, t):
        for key, value in dictionary.items():

            self.variables_df[t].loc[key] = value.flatten() if isinstance(value, np.ndarray) else value
    
    def df_to_dict(self, var, t):
        calib_dict = self.calib_var_dict if var else self.calib_par_dict
        df = self.variables_df if var else self.parameters_df
        
        shapes = {key:np.shape(value) for key, value in calib_dict.items()}
        new_dict = calib_dict.copy()

        for key in new_dict.keys():
            newvalue = np.array(df[t].loc[key].astype(float)).reshape(shapes[key]) if hasattr(df[t].loc[key] , "__len__") else (df[t].loc[key])
            new_dict[key] = newvalue

        return new_dict

    def evolve_K(self,t):
        self.parameters_df[t+1].loc['K'] = self.variables_df[t].loc['Knext']
    
    def __init__(self, Years, Growth_ratios, Variables_dict, Parameters_dict, Dynamic_parameters={}):

        self.years=Years
        self.growth_ratios = Growth_ratios
        self.calib_var_dict = Variables_dict
        self.calib_par_dict = Parameters_dict
        self.dynamic_parameters = Dynamic_parameters
        endoKnext= True if "K" not in {**self.dynamic_parameters,**self.growth_ratios}.keys() else False
        if endoKnext:
            K0=np.array([np.nan]*(len(self.years)))
            K0[0]=self.calib_par_dict['K']
            self.dynamic_parameters['K']=K0
        #GDPreal=np.array(pd.read_csv("data/GDPreal_evolution.csv")[self.years.astype(str)].iloc[0])

        #Lgrowth=np.array( pd.read_csv("data/L_growth_ratio.csv")[self.years.astype(str)].iloc[0] )
        #L=Lgrowth*[self.calib_par_dict['L']]
        
        self.dynamic_parameters={
        
            **self.evolve_par(),
            **self.dynamic_parameters
        }
        
        self.__initialize_variables_df()
        self.__initialize_parameters_df()
        if endoKnext:
            self.evolve_K(self.years[0])
        
#system = sys_df(years,growth_rates,variables,parameters)

#DeepDiff(variables,system.df_to_dict(var=True,t=2015))


# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:51:41 2020

@author: Martina Feijoo Fontán (martina.feijoo@rai.usc.es)
main .py for data analysis in STRATOS
"""
import matplotlib.pyplot      as plt
import numpy                  as np

import scipy.stats 
import sklearn.decomposition
from   sklearn.preprocessing  import StandardScaler


from plots_temperature            import *
from read_and_PCA_temperature     import *
from hist_corr_temperature        import *
from PC_temporal_evol_temperature import *

# this are files where we have saved all the functions, 
# to see further explanations about them, you can go there

plt.close('all')

#======================================================
#               YEAR SELECTION, .TXT READING
#======================================================

year=2017
filename=str(year)+'/santiago_temp_press_'+str(year)+'.txt'
file_param=str(year)+'/parametros_temperaturas'+str(year)+'.txt'

T=read_txt(filename)
cut1=147 ; cut2=153 #here we can selected a certain period (DOYs)

T_9=np.hstack((T[:,0].reshape(-1,1),T[:,2].reshape(-1,1),T[:,6].reshape(-1,1),T[:,8].reshape(-1,1),T[:,11].reshape(-1,1),T[:,15].reshape(-1,1),T[:,17].reshape(-1,1), T[:,23].reshape(-1,1), T[:,36].reshape(-1,1)))

#==========================================================
#       PERIOD SELECTION (in terms of data taken (1460))
#==========================================================

T_norm = StandardScaler().fit_transform(T) #full year  
  
#T_norm = StandardScaler().fit_transform(T[0:124]) #january
#T_norm = StandardScaler().fit_transform(T[124:236]) #february
#T_norm = StandardScaler().fit_transform(T[236:360]) #march
#T_norm = StandardScaler().fit_transform(T[360:480]) #april
#T_norm = StandardScaler().fit_transform(T[480:604]) #may
#T_norm = StandardScaler().fit_transform(T[604:724]) #june
#T_norm = StandardScaler().fit_transform(T[724:848]) #july
#T_norm = StandardScaler().fit_transform(T[848:972]) #august
#T_norm = StandardScaler().fit_transform(T[972:1092]) #september
#T_norm = StandardScaler().fit_transform(T[1092:1216]) #october
#T_norm = StandardScaler().fit_transform(T[1216:1336]) #november
#T_norm = StandardScaler().fit_transform(T[1336:]) #december

#T_norm = StandardScaler().fit_transform(T[0:236]) #1st bim
#T_norm = StandardScaler().fit_transform(T[236:480]) #2nd bim
#T_norm = StandardScaler().fit_transform(T[480:724]) #3rd bim
#T_norm = StandardScaler().fit_transform(T[724:972]) #4th bim
#T_norm = StandardScaler().fit_transform(T[972:1216]) #5th bim
#T_norm = StandardScaler().fit_transform(T[1216:]) #6th bim

#======================================================
#                     ANALYSIS
#======================================================

cmap_cut(T, year, cut1, cut2)  #temperature profile 

direction, var = PCA_37(T, T_norm)
T9, direction9, var9 = PCA_9(T)
#direction7, var7 = aemet(T)

T_normal=np.vstack((T_9[0:cut1,:] , T_9[cut2:-1, :]))
T_an=T_9[cut1:cut2, :]
    
plot_37(direction, var, year)
plot_9(direction9, var9, year)
plot_aemet(direction7, var7, year)

direction_id, var_id, direction_id31, var_id31, T_id = ideal(file_param, T)
plot_id, plot_idcut = plot_ideal(direction_id, var_id, direction_id31, var_id31, year)

T9_id=np.hstack((T_id[:,0].reshape(-1,1),T_id[:,2].reshape(-1,1),T_id[:,6].reshape(-1,1),T_id[:,8].reshape(-1,1),T_id[:,11].reshape(-1,1),T_id[:,15].reshape(-1,1),T_id[:,17].reshape(-1,1), T_id[:,23].reshape(-1,1), T_id[:,36].reshape(-1,1)))
       
# deviation and correlation analysis
T_desv_mean, sigma, k, skew = hist_analysis(T_id, T, year, cut1, cut2)
corr(T_desv_mean, sigma, k, skew, year, cut1, cut2) #correlation plots
        
PCs = PC_evolution(T9, direction_id, year, cut1, cut2) #PC (real data) evolution (time)
PCs_id = PC_evol_ideal(T9, T9_id, direction_id, year, cut1, cut2)
PCs = id_vs_real_PC(T9, direction_id, PCs_id, year, cut1, cut2)


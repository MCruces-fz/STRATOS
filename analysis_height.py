#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:13:33 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)
STRATOS
"""

import matplotlib.pyplot      as plt
import numpy                  as np

#import scipy.stats
#import scipy.optimize
import sklearn.decomposition
from   sklearn.preprocessing  import StandardScaler

from   functions_height       import * 
#this is a file where we have saved all the functions

plt.close('all')

#======================================================
#               YEAR SELECTION, .TXT READING
#======================================================

year=2014

filename=str(year)+'/santiago_height_press_'+str(year)+'.txt'
file_param=str(year)+'/parametros_temperaturas'+str(year)+'.txt'

T=read_txt(filename)
  
#======================================================
#                     PERIOD SELECTION
#======================================================

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

direction, var = PCA_37(T, T_norm)
T9, direction9, var9 = PCA_9(T)
direction7, var7 = aemet(T)
    
#plot_37(direction, var, year)
#plot_9(direction9, var9, year)
#plot_aemet(direction7, var7, year)

direction_id, var_id, direction_id31, var_id31, T_id = ideal(file_param, T)
#plot_id, plot_idcut = plot_ideal(direction_id, var_id, direction_id31, var_id31, year)

T=T/(9.8*1000) ; T_id=T_id/(9.8*1000)  

#deviation and correlation analysis
T_desv_mean, sigma, k, skew = hist_analysis(T_id, T, year)
corr(T_desv_mean, sigma, k, skew, year) #correlation plots
        
cmap(T, year)  #temperature profile             

PCs = PC_evolution(T, direction_id, year) #PC (real data) evolution (time)
PCs_id = PC_evol_ideal(T, T_id, direction_id, year)
PCs = id_vs_real_PC(T, direction_id, PCs_id, year)


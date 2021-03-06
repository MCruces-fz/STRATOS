#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:09:26 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)

in this .py we have saved all the functions which are 
exclusively temperature plots (annual profile, PCA and decompositions)
"""

import numpy              as np
import matplotlib.pyplot  as plt


def cmap_cut(T, year, cut1, cut2):
    '''
    plots temperature profile of the whole year, drawing two vertical lines
    in cut1 and cut2 to indicate a period of time
    Parameters
    ----------
    T : array 
        2D array which contains temperature data (different rows correspond to
        different times, columns are pressure levels)
    year : int
        year corresponding to the data
    cut1 : int
        DOY when we want the selected region to start (first line)
    cut2 : int
        DOY when we want the selected region to finish (second line)

    Returns
    -------
    temperature plot as colormap (red=higher T, blue=lower T)
    '''
    
    fig=plt.figure(12, figsize=(19,7))
    ax=fig.add_subplot(111)
    plt.title('Temperature profile over Santiago de Compostela  (%a) \n'
              'selected region (%i-%i) DOY' %(year, cut1, cut2), fontsize=24)
    im=ax.imshow(T.T, cmap='jet', aspect='auto', interpolation='nearest', label='T')
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    plt.yticks((np.linspace(0,37,19)), labels=np.array([1, 3,  7,  20,  50,  100,  150,  200,  250,  350,  450,  550,  650,  750,  800,  850, 900, 950,  1000]), fontsize=16)
    plt.xlabel('DOY', fontsize=20)
    plt.vlines(cut1*4, ymin=0, ymax=36, linestyle='dashed')
    plt.vlines(cut2*4, ymin=0, ymax=36, linestyle='dashed')
    ax1=ax.twinx()
    ax1.set_ylabel('Height (km)', rotation=0, fontsize=20)
    ax1.yaxis.set_label_coords(1.05, -0.05)
    plt.yticks((np.array([0,4,8,12,16,20,24,28,32,36])), labels=np.array([0, 1, 2, 3.8, 6.6, 10 ,13, 20, 32 ,45]), fontsize=15)
    ax.set_ylabel('Pressure (hPA)', fontsize=20)
    cbar=plt.colorbar(im) 
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('T (K)', rotation=0, fontsize=16)
    return


def plot_37(direction, var, year):
    '''
    Parameters
    ----------
    direction : array
        linear decomposition (coefs) of PCs in terms of levels
    var : array
        weight (normalized to 1) of the different PCs
    year : int
        year corresponding to data

    Returns
    -------
    plot: decomposition of the 4 main PCs in terms of 37 pressure levels
    vertical blue bars indicate 9 levels selected as the most relevant ones,
    they usually correspond to extremes or cuts
    '''
    
    plt.figure(1)
    plt.hlines(y=0, xmin=0, xmax=37, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a real data (1-1000)hPa' %year)
    plt.xticks(ticks=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), labels=np.array(['0','2','6','8','11','15','17','23','36']))
    plt.bar(x=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), bottom=-0.5, height=1, alpha=0.15)
    if year==2014:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), -direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2015:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2016:
        plt.plot(np.array(range(37)), direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), -direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2017:
        plt.plot(np.array(range(37)), direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2018:
        plt.plot(np.array(range(37)), direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), -direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2019:
        plt.plot(np.array(range(37)), direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    plt.legend(loc='best')
    return 


def plot_9(direction9, var9, year):
    '''
    same as plot_37 but now we took 9 levels selected there and made 
    the PCA just with them (as they are 'relevant' PCs must look similar)

    '''
    plt.figure(2)
    plt.hlines(y=0, xmin=0, xmax=36, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a (9 lvls)' %year)
    plt.xticks(ticks=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), labels=np.array(['0','2','6','8','11','15','17','23','36']))
    if year==2014:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2015:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2016:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2017:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2018:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2019:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    plt.legend()
    return 


def plot_aemet(direction7, var7, year):
    '''
    same as the two previus ones but here 7 levels corresponding to the ones 
    studied in AEMET were selected

    '''
    plt.figure(3)
    plt.hlines(y=0, xmin=0, xmax=37, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a real data (AEMET)' %year)
    plt.bar(x=np.array([6,8,12,14,17,21,36]), bottom=-0.8, height=1.4, alpha=0.15)
    plt.xticks(ticks=np.array([6,8,12,14,17,21,36]), labels=np.array(['6','8','12','14','17','21','36']))
    
    if year==2014:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2015:
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2016:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2017:
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2018:
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2019:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    plt.legend(loc='best')
    return 


def plot_ideal(direction_id, var_id, direction_id31, varid31, year):
    '''
    Parameters
    ----------
    direction_id : array
        linear decomposition of ideal PCs in terms of levels. 
        to see how ideal PCs are calculated go to read_and_PCA_temperature.py
        
    var_id : array
        weights of each PC
        
    direction_id31 : array
        linear decomposition of ideal PCs in terms of levels,
        but without considering the first 6 (we did this just to compare with AEMET)
        
    varid31 : array
        weights of each PC
    year : int

    Returns
    -------
    linear decomposition plot of PCs in terms of pressure levels

    '''
    plt.figure(4)
    plt.hlines(y=0, xmin=0, xmax=37, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCs %a ideal data (1-1000)hPa' %year)
    plt.bar(x=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), bottom=-0.5, height=1, alpha=0.15, color='red')
    plt.xticks(ticks=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), labels=np.array(['0','2','6','8','11','15','17','23','36']))
    if year==2014:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2015:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), -direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2016:
        plt.plot(np.array(range(37)), direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), -direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2017:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2018:
        plt.plot(np.array(range(37)), direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2019:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    plt.legend()
    
    plt.figure(5)
    plt.hlines(y=0, xmin=0, xmax=37, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a ideal data (20-1000)hPa' %year)
    plt.bar(x=np.array([6, 8, 11, 14, 16, 24, 36]), bottom=-0.5, height=1, alpha=0.15, color='red')
    plt.xticks(ticks=np.array([6, 8, 11, 15, 17, 23, 36]), labels=np.array(['6','8','11','15','17','23','36']))
    if year==2014:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2015:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2016:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2017:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2018:
        plt.plot(np.array(range(6,37)), direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2019:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    plt.legend()

    return plt.figure(4), plt.figure(5)

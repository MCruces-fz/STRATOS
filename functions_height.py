#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:15:10 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)
STRATOS
"""
import numpy as np
#import os
#os.environ['PROJ_LIB'] = r'C:\Users\Martina\anaconda3\pkgs\basemap-1.3.0-py38ha7665c8_0\Library\share\basemap'
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn.decomposition
import matplotlib.pyplot as plt
#import scipy.stats 
import matplotlib.axes
#import scipy.optimize
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew
from matplotlib.colors import LogNorm

def read_txt(file_name):
    InFile = open(file_name,'r+')
    lines=InFile.readlines()
    Pressure=[]
    Temperature=np.zeros(((len(lines)), 37))
    plist=lines[0].strip().split(' ')
        
    for p in plist[1:]:
        Pressure.append(float(p))
        Days=[]
        Month=[1.]
        Day=[1.]
        days=0
        for i in range(len(lines))[1:]:
            line=lines[i]
            date= line.strip().split(' ')[0]
            month=float(date.strip().split('-')[1])
            day=float( date.strip().split('-')[2])
            Month.append(month)
            Day.append(day)
            if Day[i]>=Day[i-1]:
                if Month[i]==Month[i-1]:
                    days=days+(Day[i]-Day[i-1])
       
            if day==1. and Day[i-1]>1:
                days=days+1
            time= line.strip().split(' ')[1] 
            hour=float(time.strip().split(':')[0])
            minutes=float(time.strip().split(':')[1])
            seconds=float(time.strip().split(':')[2][:-1])
            time=seconds+60*minutes+60*60*hour+24*60*60*(days)
            Days.append(time/(60*60*24))
            
            temp= line.strip().split(' ')[2:]
            
            for j in range(37):
                Temperature[i,j]=float(temp[j]) 
                
    return Temperature[1:]


def PCA_37(T, T_norm):
    scaler = StandardScaler().fit(T)
    comp=37 #number of components (target)
    info=0.8 #insert amount of info we'd like to keep (info/1)
    pca_temp =sklearn.decomposition.PCA(n_components=comp)
    principalComponents_temp = pca_temp.fit_transform(T_norm)
    principal_temp_Df = pd.DataFrame(data = principalComponents_temp
                 , columns = ['PC ' + str(i) for i in range(1,comp+1) ])
    var=pca_temp.explained_variance_ratio_
    
    #print ('lost info when reducing to', comp, 'components=', (1-sum(var))*100, '%')
    
    #the principal components are defined by:
        
    direction=pca_temp.components_ #components of the PC's in terms of the initial 37 levels
    lenght=pca_temp.explained_variance_
    decomp_temp_Df = pd.DataFrame(data = direction
                 , columns = ['lvl ' + str(i) for i in range(1,comp+1) ])
    
    #searching for the number of PC's we need
    s=0
    count=0
    if var[0]<info:
        while s<info:
            s=s+var[count] 
            count+=1
    else:
        s=var[0]

    print (count, 'PCs needed for keeping', int(info*100), '% of info', '(',"{:.2f}".format(s*100), '%)')
    return direction, var

def PCA_9(y):
    y9=np.hstack((y[:,0].reshape(-1,1),y[:,2].reshape(-1,1),y[:,6].reshape(-1,1),y[:,8].reshape(-1,1),y[:,11].reshape(-1,1),y[:,15].reshape(-1,1),y[:,17].reshape(-1,1), y[:,23].reshape(-1,1), y[:,36].reshape(-1,1)))
    #y9=np.hstack((y_norm[:,0].reshape(-1,1),y_norm[:,2].reshape(-1,1),y_norm[:,6].reshape(-1,1),y_norm[:,8].reshape(-1,1),y_norm[:,11].reshape(-1,1),y_norm[:,15].reshape(-1,1),y_norm[:,17].reshape(-1,1), y_norm[:,23].reshape(-1,1), y_norm[:,36].reshape(-1,1)))
    y9_norm = StandardScaler().fit_transform(y9) 
    scaler9 = sklearn.preprocessing.StandardScaler().fit(y9)
    y9=scaler9.transform(y9)
    pca_temp9 = sklearn.decomposition.PCA(n_components=9)
    principalComponents_temp9 = pca_temp9.fit_transform(y9_norm)
    direction9=pca_temp9.components_
    loading_scores9 = pd.Series(pca_temp9.components_[0])
    sorted_loading_scores9 = loading_scores9.abs().sort_values(ascending=False)
    var9=pca_temp9.explained_variance_ratio_
    return y9, direction9, var9
  
def plot_37(direction, var, year):
    plt.figure(1)
    plt.hlines(y=0, xmin=0, xmax=37, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a real data (1-1000)hPa' %year)
    plt.xticks(ticks=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), labels=np.array(['0','2','6','8','11','15','17','23','36']))
    plt.bar(x=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), bottom=-0.5, height=1, alpha=0.15)
    if year==2014:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), -direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2015:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2016:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2017:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), -direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2018:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    if year==2019:
        plt.plot(np.array(range(37)), -direction[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var[0]))+ ')')
        plt.plot(np.array(range(37)), -direction[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var[1]))+ ')')
        plt.plot(np.array(range(37)), -direction[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var[2]))+ ')')
        plt.plot(np.array(range(37)), direction[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var[3]))+ ')')
    plt.legend(loc='best')
    return 

def plot_9(direction9, var9, year):
    plt.figure(2)
    plt.hlines(y=0, xmin=0, xmax=36, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a (9 lvls)' %year)
    plt.xticks(ticks=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), labels=np.array(['0','2','6','8','11','15','17','23','36']))
    plt.bar(x=np.array([0,2,6, 8, 11, 15, 17, 23, 36]), bottom=-0.5, height=1.2, alpha=0.15)
    if year==2014:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2015:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2016:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2017:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2018:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    if year==2019:
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var9[0]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var9[1]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), -direction9[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var9[2]))+ ')')
        plt.plot(np.array([0,2,6, 8, 11, 15, 17, 23, 36]), direction9[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var9[3]))+ ')')
    plt.legend()
    return 

def aemet(y):
    '''
    PC analysis choosing the levels aemet uses
    
    Parameters
    ----------
    y : array
        which contains temperature data

    Returns
    -------
    direction7 : array
        PC decomposition in terms of the 7 AEMET levels
    var7 : array
        weights of each PC

    '''
    y7=np.hstack((y[:,6].reshape(-1,1),y[:,8].reshape(-1,1),y[:,12].reshape(-1,1),y[:,14].reshape(-1,1),y[:,17].reshape(-1,1),y[:,21].reshape(-1,1),y[:,36].reshape(-1,1)))
    y7_norm = StandardScaler().fit_transform(y7) #full year
    scaler7 = sklearn.preprocessing.StandardScaler().fit(y7)
    y_7=scaler7.transform(y7)
    pca = sklearn.decomposition.PCA()
    pca_temp7 = sklearn.decomposition.PCA(n_components=7)
    principalComponents_temp7 = pca_temp7.fit_transform(y7_norm)
    direction7=pca_temp7.components_
    loading_scores7 = pd.Series(pca_temp7.components_[0])
    var7=pca_temp7.explained_variance_ratio_
    pca.fit(y_7)
    X7=np.dot(pca.transform(y_7)[:,:4], pca.components_[:4,:])
    inversa=scaler7.inverse_transform(X7)
    dif7=y7-inversa
    
    return direction7, var7

def plot_aemet(direction7, var7, year):
    plt.figure(3)
    plt.hlines(y=0, xmin=0, xmax=37, linestyle='dotted', color='black', linewidth=0.5)
    plt.title('PCA %a real data (AEMET)' %year)
    plt.bar(x=np.array([6,8,12,14,17,21,36]), bottom=-0.8, height=1.4, alpha=0.15)
    plt.xticks(ticks=np.array([6,8,12,14,17,21,36]), labels=np.array(['6','8','12','14','17','21','36']))
    
    if year==2014:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2015:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2016:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2017:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2018:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    elif year==2019:
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var7[0]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var7[1]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), -direction7[2], '-', linestyle='dashed', alpha=0.8, linewidth=1.2, label='PC3 ('+ str("{:.2f}".format(var7[2]))+ ')')
        plt.plot(np.array([6,8,12,14,17,21,36]), direction7[3], '-', linestyle='dashed', alpha=0.7, linewidth=0.6, label='PC4 ('+ str("{:.2f}".format(var7[3]))+ ')')
    plt.legend(loc='best')
    return 


def ideal(file_param, T):
    '''
    PCA of ideal data taking in account the parameters in file_param
    these parameters are: amplitude (A), phase
    we suppose ideal data ~A*np.sin(x/T-phase)

    the part where we use just 31 levels instead of 37 is the same as the first one,
    but we 'cut' the first 6 because we wanted to compare with AEMET
    and they don't have data around that levels'
    
    Parameters
    ----------
    file_param : str 
        (.txt) where we have fitted curve params
    T : array
        temperature data

    Returns
    -------
    direction_id: array
        PC decomposition of ideal data (sin curve)
    var_id: array
        weights of each PC
    direction_id31: array
        PC decomposition of ideal data (sin curve)
    var_id31: array
        weights of each PC

    '''
    InFile = open(file_param,'r+')
    lines_param=InFile.readlines()
    A=np.zeros(len(lines_param)-1); phases=np.zeros(len(lines_param)-1)
    for i in range(len(lines_param)-1):
        wi=lines_param[i+1].split(',')
        A[i]=np.float(wi[0])
        phases[i]=np.float(wi[1])
    
    def f(A, x, phases):
        return A*np.sin(2*np.pi*x/len(T) - phases)
    
    ideal=np.zeros((37,len(T)))
    x=np.arange(0,len(T),1)
    for i in range(37):
        ideal[i]=f(A[i], x, phases[i])*9.8*1000+np.median(T[:,i])
        
    y_ideal=StandardScaler().fit_transform(ideal.T) 
    pca_ideal = sklearn.decomposition.PCA(n_components=37)
    principalComponents_ideal = pca_ideal.fit_transform(y_ideal)
    direction_id=pca_ideal.components_
    var_id=pca_ideal.explained_variance_ratio_
    
    ideal_31=np.zeros((len(T), 31))
    for i in range(31):
        ideal_31[:,i]=y_ideal[:,i+6]
        
    y_ideal_31=StandardScaler().fit_transform(ideal_31) 
    pca_ideal = sklearn.decomposition.PCA(n_components=31)
    principalComponents_ideal = pca_ideal.fit_transform(y_ideal_31)
    direction_id31=pca_ideal.components_
    var_id31=pca_ideal.explained_variance_ratio_

    return direction_id, var_id, direction_id31, var_id31, ideal.T

def plot_ideal(direction_id, var_id, direction_id31, varid31, year):
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
        plt.plot(np.array(range(37)), direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2016:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2017:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), -direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2018:
        plt.plot(np.array(range(37)), -direction_id[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(37)), -direction_id[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
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
        plt.plot(np.array(range(6,37)), direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2015:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2016:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2017:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2018:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    elif year==2019:
        plt.plot(np.array(range(6,37)), -direction_id31[0], '-', linestyle='dashed', linewidth=2.5, label='PC1 ('+ str("{:.2f}".format(var_id[0]))+ ')')
        plt.plot(np.array(range(6,37)), -direction_id31[1], '-', linestyle='dashed', alpha=0.9, linewidth=1.6, label='PC2 ('+ str("{:.2f}".format(var_id[1]))+ ')')
    plt.legend()

    return plt.figure(4), plt.figure(5)

def hist_analysis(T_ideal, T, year):
    '''
    statistical analysis of deviations between T and T_ideal

    Parameters
    ----------
    T_ideal : array
        ideal temperatures (calculated before as np.sin from file_param)
    T : array
        real temperatures
    year : int
        DESCRIPTION.

    Returns
    -------
    T_desv_mean : array
        mean of differences between real data and ideal data 
        calculated for the 9 levels we selected before
    sigma : array
        standard deviation of the analyzed data (2nd moment)
    k : array
        kurtosis (3rd moment)
    skew : array
        skewness (4th moment)

    '''
    T=np.hstack((T[:,0].reshape(-1,1),T[:,2].reshape(-1,1),T[:,6].reshape(-1,1),T[:,8].reshape(-1,1),T[:,11].reshape(-1,1),T[:,15].reshape(-1,1),T[:,17].reshape(-1,1), T[:,23].reshape(-1,1), T[:,36].reshape(-1,1)))
    T_ideal=np.hstack((T_ideal[:,0].reshape(-1,1),T_ideal[:,2].reshape(-1,1),T_ideal[:,6].reshape(-1,1),T_ideal[:,8].reshape(-1,1),T_ideal[:,11].reshape(-1,1),T_ideal[:,15].reshape(-1,1),T_ideal[:,17].reshape(-1,1), T_ideal[:,23].reshape(-1,1), T_ideal[:,36].reshape(-1,1)))
    T_desv=np.zeros((len(T_ideal), len(T_ideal[0])))
    T_desv_mean=np.zeros(len(T_ideal))
    sigma=np.zeros(len(T_ideal))
    k=np.zeros(len(T_ideal))
    sk=np.zeros(len(T_ideal))
    for i in range(len(T_ideal)): #selecting time
        for j in range(len(T_ideal[i])): #selecting level
            T_desv[i,j]=T[i,j]-T_ideal[i,j]
        T_desv_mean[i]=np.mean(T_desv[i])
        sigma[i]=np.std(T_desv[i])
        k[i]=kurtosis(T_desv[i], fisher=True)
        sk[i]=skew(T_desv[i])
    
    def parabolic(x,a,b,c):
        return a*x**2+b*x+c
    
    def pol4(x,a,b,c,d,e):
        return a*x**4+b*x**3+c*x**2+d*x+e
    
    fit, cov=curve_fit(parabolic, np.array(range(len(T))), T_desv_mean)
    fit1, cov1=curve_fit(pol4, np.array(range(len(T))), T_desv_mean)
    
    T_desv_corr=np.zeros(len(T_ideal))
    T_desv_corr2=np.zeros(len(T_ideal))
    sigma_corr=np.zeros(len(T_ideal))
    k_corr=np.zeros(len(T_ideal))
    skew_corr=np.zeros(len(T_ideal))
    for i in range(len(T_ideal)): #selecting time
        T_desv_corr2[i]=T_desv_mean[i]-parabolic(np.array(range(len(T_ideal))), *fit)[i]
        T_desv_corr[i]=T_desv_mean[i]-pol4(np.array(range(len(T_ideal))), *fit1)[i]
        sigma_corr[i]=np.std(T_desv_corr[i])
        k_corr[i]=kurtosis(T_desv_corr[i], fisher=True)
        skew_corr[i]=skew(T_desv_corr[i])
    
    '''plt.figure(88)
    plt.hlines(0,0, 366, linewidth=0.8)
    plt.plot(np.arange(0,len(k),1)/4., T_desv_mean, label='mean', linewidth=1.)
    plt.plot(np.arange(0,len(k),1)/4., T_desv_corr, label='corr pol 4th grade', linewidth=1.)
    plt.plot(np.arange(0,len(k),1)/4., T_desv_corr2, label='corr pol 2nd grade', linewidth=1.)
    plt.legend()'''
    
    fig=plt.figure(6)
    plt.suptitle(year, fontsize=24)
    fig.subplots_adjust (hspace=0.9)
   
    ax1=plt.subplot(411)
    plt.plot(np.arange(0,len(k),1)/4., T_desv_mean)
    plt.xticks(ticks=np.array(range(0,len(k),200)), fontsize=15)
    ax1.set_title('Deviations (mean)', fontsize=16)
    ax1.set_xlabel('DOY', fontsize=16)
    ax1.set_xticks(np.array([0, 50, 100, 150, 200, 250, 300, 350]))
    ax1.xaxis.set_label_coords(1., -0.2)
    ax1.set_ylabel('Height (km)', rotation=0, fontsize=16)
    ax1.yaxis.set_label_coords(0., 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.hlines(0, 0, 365, linewidth=0.6, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xlim(0,370)
    plt.legend(loc='best', ncol=3)
    
    ax2=plt.subplot(412)
    plt.plot(np.arange(0,len(k),1)/4., sigma, label='standard dev', color='orange')
    plt.xticks(ticks=np.array(range(0,len(k),200)), fontsize=15)
    ax2.set_title('Standard deviation', fontsize=16)
    ax2.set_xlabel('DOY', fontsize=16)
    ax2.xaxis.set_label_coords(1., -0.2)
    ax2.set_ylabel('Height (km)', rotation=0, fontsize=16)
    ax2.yaxis.set_label_coords(0., 1.1)
    ax2.set_xticks(np.array([0, 50, 100, 150, 200, 250, 300, 350]))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #plt.hlines(0, 0, 365, linewidth=0.6, linestyle='--')
    plt.ylim(0, np.max(sigma))
    plt.yticks(fontsize=15)
    plt.xlim(0,370)
    
    ax3=plt.subplot(413)
    plt.plot(np.arange(0,len(k),1)/4., k, label='kurtosis', color='green')
    plt.xticks(ticks=np.array(range(0,len(k),200)), fontsize=15)
    ax3.set_title('Skewness (A)', fontsize=16)
    ax3.set_xlabel('DOY', fontsize=16)
    ax3.xaxis.set_label_coords(1., -0.2)
    ax3.set_ylabel('A', rotation=0, fontsize=16)
    ax3.yaxis.set_label_coords(0., 1.1)
    ax3.set_xticks(np.array([0, 50, 100, 150, 200, 250, 300, 350]))
    plt.yticks(fontsize=15)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.hlines(0, 0, 365, linewidth=0.6, linestyle='--')
    plt.xlim(0,370)
    
    ax4=plt.subplot(414)
    plt.plot(np.arange(0,len(k),1)/4., sk, label='skewness', color='red')
    ax4.set_title('Kurtosis (K)', fontsize=16)
    ax4.set_xlabel('DOY', fontsize=16)
    ax4.xaxis.set_label_coords(1., -0.2)
    ax4.set_ylabel('K', rotation=0, fontsize=16)
    ax4.yaxis.set_label_coords(0., 1.1)
    ax4.spines['top'].set_visible(False)
    plt.xticks((np.array([0, 50, 100, 150, 200, 250, 300, 350])), fontsize=15)
    plt.yticks(fontsize=15)
    ax4.spines['right'].set_visible(False)
    plt.hlines(0, 0, 365, linewidth=0.6, linestyle='--')
    plt.xlim(0,370)
    
    return T_desv_mean, sigma, k, sk

def corr(T_desv_mean, sigma, k, skew, year):
    '''

    Parameters
    ----------
    T_desv_mean : array
        deviation of temperatures (real minus ideal) for 9 levels
        mean of this 9 deviation values
    sigma : array
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    skew : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure(7)
    plt.suptitle('Correlations with mean (%a)' %year , fontsize=24)
    ax1=plt.subplot(121)
    plt.xlabel('Deviations (mean)', fontsize=24)
    plt.plot(T_desv_mean, k, '.', label='kurtosis')
    plt.ylabel('Kurtosis', fontsize=24)
    plt.yticks(fontsize=15)
    plt.hlines(0, np.min(T_desv_mean), np.max(T_desv_mean),  linewidth=0.6, linestyle='--')
    plt.vlines(0, np.min(k), np.max(k), linewidth=0.6, linestyle='--')
    plt.xticks(fontsize=15)
    
    ax2=plt.subplot(122)
    plt.xlabel('Deviations (mean)', fontsize=24)
    plt.plot(T_desv_mean, skew, '.', label='skewness')
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.hlines(0, np.min(T_desv_mean), np.max(T_desv_mean),  linewidth=0.6, linestyle='--')
    plt.vlines(0, np.min(skew), np.max(skew), linewidth=0.6, linestyle='--')
    plt.ylabel('Skewness', fontsize=24)
    
    plt.figure(8)
    plt.suptitle('Correlation with standard deviation (%a)' %year , fontsize=24)
    ax1=plt.subplot(121)
    plt.xlabel('Standard deviation', fontsize=24)
    plt.plot(sigma, k, '.', label='kurtosis')
    plt.ylabel('Kurtosis', fontsize=24)
    plt.hlines(0, np.min(sigma), np.max(sigma),  linewidth=0.6, linestyle='--')
    plt.xlim(0, np.max(sigma))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    ax2=plt.subplot(122)
    plt.plot(sigma, skew, '.', label='skewness')
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlim(0, np.max(sigma))
    plt.hlines(0, np.min(sigma), np.max(sigma),  linewidth=0.6, linestyle='--')
    plt.vlines(0, np.min(skew), np.max(skew), linewidth=0.6, linestyle='--')
    plt.ylabel('Skewness', fontsize=24)
    plt.xlabel('Standard deviation', fontsize=24)
    
    plt.figure(10)
    plt.title('Correlation with skewness (%a)' %year , fontsize=24)
    plt.xlabel('Skewness ', fontsize=24)
    plt.plot(skew, k, '.', label='kurtosis')
    plt.ylabel('Kurtosis', fontsize=24)
    plt.hlines(0, np.min(skew), np.max(skew),  linewidth=0.6, linestyle='--')
    plt.vlines(0, np.min(k), np.max(k), linewidth=0.6, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    return 
  
def cmap(T, year):
    fig=plt.figure(12, figsize=(19,7))
    ax=fig.add_subplot(111)
    plt.title('Heights profile over Santiago de Compostela  (%a)' %year, fontsize=24)
    im=ax.imshow(T.T, cmap='jet', aspect='auto', interpolation='nearest', label='T', norm=LogNorm(vmin=1, vmax=55))
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    plt.yticks((np.linspace(0,37,19)), labels=np.array([1, 3,  7,  20,  50,  100,  150,  200,  250,  350,  450,  550,  650,  750,  800,  850, 900, 950,  1000]), fontsize=16)
    plt.xlabel('DOY', fontsize=20)
    ax1=ax.twinx()
    ax1.set_ylabel('Height (km)', rotation=0, fontsize=20)
    ax1.yaxis.set_label_coords(1.05, -0.05)
    plt.yticks((np.array([0,4,8,12,16,20,24,28,32,36])), labels=np.array([0, 1, 2, 3.8, 6.6, 10 ,13, 20, 32 ,45]), fontsize=15)
    ax.set_ylabel('Pressure (hPA)', fontsize=20)
    cbar=plt.colorbar(im) 
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r'Height (km)', 0, fontsize=16, labelpad=10)
    return

def PC_evolution(T, direction, year):
    T9=np.hstack((T[:,0].reshape(-1,1),T[:,2].reshape(-1,1),T[:,6].reshape(-1,1),T[:,8].reshape(-1,1),T[:,11].reshape(-1,1),T[:,15].reshape(-1,1),T[:,17].reshape(-1,1), T[:,23].reshape(-1,1), T[:,36].reshape(-1,1)))
    dir9=np.hstack((direction[:,0].reshape(-1,1),direction[:,2].reshape(-1,1),direction[:,6].reshape(-1,1),direction[:,8].reshape(-1,1),direction[:,11].reshape(-1,1),direction[:,15].reshape(-1,1),direction[:,17].reshape(-1,1), direction[:,23].reshape(-1,1), direction[:,36].reshape(-1,1)))

    PCs=np.zeros((len(T9), 4))
    for i in range(len(T9)):
        for j in range(4):
            PCs[i,j]=np.sum(T9[i]*dir9[j])/np.sum(dir9[j])
        
    fig=plt.figure(13)
    plt.suptitle(year, fontsize=24)
    fig.subplots_adjust (hspace=0.9)
    ax1=plt.subplot(411)
    ax1.plot(np.arange(len(T)), PCs[:,0], label='PC1') 
    ax1.set_title('PC1', fontsize=16, color='tab:blue')
    ax1.set_xlabel('DOY', fontsize=16)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax1.xaxis.set_label_coords(1., -0.2)
    ax1.set_ylabel('Height (km)', rotation=0, fontsize=16)
    ax1.yaxis.set_label_coords(0., 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,0]), 0, len(T), linewidth=0.5, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xlim(0,len(T))
    
    ax2=plt.subplot(412)
    plt.plot(np.arange(len(T)), PCs[:,1], label='PC2', color='orange')
    ax2.set_title('PC2', fontsize=16, color='orange')
    ax2.set_xlabel('DOY', fontsize=16)
    ax2.xaxis.set_label_coords(1., -0.2)
    ax2.yaxis.set_label_coords(0., 1.1)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,1]), 0, len(T), linewidth=0.5, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xlim(0,len(T))
    
    ax3=plt.subplot(413)
    plt.plot(np.arange(len(T)), PCs[:,2], label='PC3', color='green')
    plt.yticks(fontsize=15)
    ax3.set_title('PC3', fontsize=16, color='green')
    ax3.set_xlabel('DOY', fontsize=16)
    ax3.xaxis.set_label_coords(1., -0.2)
    ax3.yaxis.set_label_coords(0., 1.1)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,2]), 0, len(T), linewidth=0.5, linestyle='--')
    plt.xlim(0,len(T))
    
    ax4=plt.subplot(414)
    plt.plot(np.arange(len(T)), PCs[:,3], label='PC4', color='red')
    ax4.set_title('PC4', fontsize=16, color='red')
    ax4.set_xlabel('DOY', fontsize=16)
    ax4.xaxis.set_label_coords(1., -0.2)
    plt.yticks(fontsize=15)
    ax4.yaxis.set_label_coords(0., 1.1)
    ax4.spines['top'].set_visible(False)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax4.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,3]), 0, len(T), linewidth=0.5, linestyle='--')
    plt.xlim(0,len(T))
    return PCs

def PC_evol_ideal(T, T_id, direction_id, year):
    T_9=np.hstack((T[:,0].reshape(-1,1),T[:,2].reshape(-1,1),T[:,6].reshape(-1,1),T[:,8].reshape(-1,1),T[:,11].reshape(-1,1),T[:,15].reshape(-1,1),T[:,17].reshape(-1,1), T[:,23].reshape(-1,1), T[:,36].reshape(-1,1)))
    T9_id=np.hstack((T_id[:,0].reshape(-1,1),T_id[:,2].reshape(-1,1),T_id[:,6].reshape(-1,1),T_id[:,8].reshape(-1,1),T_id[:,11].reshape(-1,1),T_id[:,15].reshape(-1,1),T_id[:,17].reshape(-1,1), T_id[:,23].reshape(-1,1), T_id[:,36].reshape(-1,1)))
    dir9_id=np.hstack((direction_id[:,0].reshape(-1,1),direction_id[:,2].reshape(-1,1),direction_id[:,6].reshape(-1,1),direction_id[:,8].reshape(-1,1),direction_id[:,11].reshape(-1,1),direction_id[:,15].reshape(-1,1),direction_id[:,17].reshape(-1,1), direction_id[:,23].reshape(-1,1), direction_id[:,36].reshape(-1,1)))
    
    PCid1=np.zeros((len(T9_id)))
    for i in range(len(T9_id)):
        for j in range(9):
            PCid1[i]=sum(dir9_id[0]*T9_id[i])/sum(dir9_id[0])
            
    PCid2=np.zeros((len(T9_id)))
    for i in range(len(T9_id)):
        for j in range(9):
            PCid2[i]= sum(dir9_id[1]*T9_id[i])/sum(dir9_id[1])
            
    PCs_id=np.zeros((len(T),2))
    PCs_id[:,0]=PCid1 ; PCs_id[:,1]=PCid2
    
    '''fig=plt.figure(14)
    plt.suptitle(year, fontsize=24)
    fig.subplots_adjust (hspace=0.9)
    ax1=plt.subplot(211)
    plt.plot(np.arange(len(T_id)), PCs_id[:,0], label='PC1')
    ax1.set_title('PC1', fontsize=16, color='tab:blue')
    ax1.set_xlabel('DOY', fontsize=16)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax1.xaxis.set_label_coords(1., -0.2)
    ax1.set_ylabel('T (K)', rotation=0, fontsize=16)
    ax1.yaxis.set_label_coords(0., 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs_id[:,0]), 0, len(T_id), linewidth=0.6, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xlim(0,len(T_id))
    
    ax2=plt.subplot(212)
    plt.plot(np.arange(len(T_id)), PCs_id[:,1], label='PC2', color='orange')
    ax2.set_title('PC2', fontsize=16, color='orange')
    ax2.set_xlabel('DOY', fontsize=16)
    ax2.xaxis.set_label_coords(1., -0.2)
    ax2.yaxis.set_label_coords(0., 1.1)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs_id[:,1]), 0, len(T_id), linewidth=0.6, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xlim(0,len(T_id))'''
    return PCs_id
    
def id_vs_real_PC(T, direction_id, PCs_id, year):
    T9=np.hstack((T[:,0].reshape(-1,1),T[:,2].reshape(-1,1),T[:,6].reshape(-1,1),T[:,8].reshape(-1,1),T[:,11].reshape(-1,1),T[:,15].reshape(-1,1),T[:,17].reshape(-1,1), T[:,23].reshape(-1,1), T[:,36].reshape(-1,1)))
    dir9_id=np.hstack((direction_id[:,0].reshape(-1,1),direction_id[:,2].reshape(-1,1),direction_id[:,6].reshape(-1,1),direction_id[:,8].reshape(-1,1),direction_id[:,11].reshape(-1,1),direction_id[:,15].reshape(-1,1),direction_id[:,17].reshape(-1,1), direction_id[:,23].reshape(-1,1), direction_id[:,36].reshape(-1,1)))
    
    PC1=np.zeros((len(T9)))
    for i in range(len(T9)):
        for j in range(9):
            PC1[i]=sum(dir9_id[0]*T9[i])/sum(dir9_id[0])
            
    PC2=np.zeros((len(T9)))
    for i in range(len(T9)):
        for j in range(9):
            PC2[i]= sum(dir9_id[1]*T9[i])/sum(dir9_id[1])
            
    PCs=np.zeros((len(T),2))
    PCs[:,0]=PC1 ; PCs[:,1]=PC2

    fig=plt.figure(15)
    fig.subplots_adjust (hspace=0.7)
    
    ax1=plt.subplot(211)
    plt.yticks(fontsize=15)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    plt.xlim(0,len(PCs[:,0]))
    plt.xlabel('DOY', fontsize=15)
    ax1.xaxis.set_label_coords(1, -0.15)
    plt.ylabel('Height (km)', rotation='horizontal', fontsize=15)
    ax1.yaxis.set_label_coords(0, 1.1)
    
    ax3=plt.subplot(212)
    plt.yticks(fontsize=15)
    plt.xlim(0,len(PCs[:,0]))
    plt.xlabel('DOY', fontsize=15)
    ax3.xaxis.set_label_coords(1, -0.15)
    
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax1.hlines(y=np.mean(PCs[:,0]), xmin=0, xmax=len(PCs[:,0]), linestyle='dotted', color='black', linewidth=0.8, alpha=0.7)
    ax3.hlines(0, xmin=0, xmax=len(PCs[:,0]), linestyle='dotted', color='black', linewidth=0.8, alpha=0.7)
    ax1.hlines(y=np.mean(PCs[:,1]), xmin=0, xmax=len(PCs[:,0]), linestyle='dotted', color='black', linewidth=0.8, alpha=0.7)
    ax1.set_title('PCA %a real data (1-1000)hPa' %year, fontsize=20)
    ax3.set_title(r'$\Delta PC= PC_{real}-PC_{ideal}$', fontsize=20)

    ax1.plot(np.array(range(len(PCs[:,0]))), PCs[:,0], '-', linewidth=1.5, label='PC1', color='deepskyblue')
    ax1.plot(np.array(range(len(PCs[:,1]))), PCs[:,1], '-', linewidth=1.5, label='PC2', color='darkorange')
    ax1.plot(np.array(range(len(PCs_id[:,0]))), PCs_id[:,0], '--', linewidth=1.5, label='PC1 ideal', color='red')
    ax1.plot(np.array(range(len(PCs_id[:,1]))), PCs_id[:,1], '--', linewidth=1.5, label='PC1 ideal', color='green')
    
    ax3.plot(np.array(range(len(PCs[:,0]))), PCs[:,0]-PCs_id[:,0], '-',  linewidth=1.5, label=r'$\Delta PC1$', c='deepskyblue')
    ax3.plot(np.array(range(len(PCs[:,1]))), PCs[:,1]-PCs_id[:,1], '-',  linewidth=1.5, label=r'$\Delta PC2$', c='darkorange')
    
    ax1.legend(loc='best', fontsize=15, ncol = 4)
    ax3.legend(loc='best', fontsize=15)
    
    plt.figure(16)
    plt.title('Correlation between fluctuations in PC1, PC2 (%a)' %year, fontsize=24)
    plt.plot(PCs_id[:,0]-PCs[:,0], PCs_id[:,1]-PCs[:,1], '.')
    plt.xlabel(r'$\Delta PC1$', fontsize=20)
    plt.ylabel(r'$\Delta PC2$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.hlines(0, np.min(PCs_id[:,0]-PCs[:,0]), np.max(PCs_id[:,0]-PCs[:,0]), linestyle='dashed')
    plt.vlines(0, np.min(PCs_id[:,1]-PCs[:,1]), np.max(PCs_id[:,1]-PCs[:,1]), linestyle='dashed')
    
    return PCs



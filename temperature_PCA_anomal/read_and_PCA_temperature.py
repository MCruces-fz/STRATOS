#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:13:56 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn.decomposition
import numpy as np

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
        ideal[i]=f(A[i], x, phases[i])
        
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

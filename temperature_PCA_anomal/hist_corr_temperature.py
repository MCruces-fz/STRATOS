#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:18:33 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.stats

def hist_analysis(T_ideal, T, year, cut1, cut2):
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
    T_ideal=np.hstack((T_ideal[:,0].reshape(-1,1),T_ideal[:,2].reshape(-1,1),T_ideal[:,6].reshape(-1,1),T_ideal[:,8].reshape(-1,1),T_ideal[:,11].reshape(-1,1),T_ideal[:,15].reshape(-1,1),T_ideal[:,17].reshape(-1,1), T_ideal[:,23].reshape(-1,1), T_ideal[:,36].reshape(-1,1)))
    
    T_desv=np.zeros((len(T_ideal), len(T_ideal[0])))
    T_desv_mean=np.zeros(len(T_ideal))
    sigma=np.zeros(len(T_ideal))
    k=np.zeros(len(T_ideal))
    skew=np.zeros(len(T_ideal))
    for i in range(len(T_ideal)): #selecting time
        for j in range(len(T_ideal[i])): #selecting level
            T_desv[i,j]=T[i,j]-T_ideal[i,j]-np.mean(T[:,j])
        T_desv_mean[i]=np.mean(T_desv[i])
        sigma[i]=np.std(T_desv[i])
        k[i]=scipy.stats.kurtosis(T_desv[i], fisher=True)
        skew[i]=scipy.stats.skew(T_desv[i])
    
    def parabolic(x,a,b,c):
        return a*x**2+b*x+c
    
    def pol4(x,a,b,c,d,e):
        return a*x**4+b*x**3+c*x**2+d*x+e
    
    fit, cov=scipy.optimize.curve_fit(parabolic, np.array(range(len(T))), T_desv_mean)
    fit1, cov1=scipy.optimize.curve_fit(pol4, np.array(range(len(T))), T_desv_mean)
    
    T_desv_corr=np.zeros(len(T_ideal))
    T_desv_corr2=np.zeros(len(T_ideal))
    sigma_corr=np.zeros(len(T_ideal))
    k_corr=np.zeros(len(T_ideal))
    skew_corr=np.zeros(len(T_ideal))
    for i in range(len(T_ideal)): #selecting time
        T_desv_corr2[i]=T_desv_mean[i]-parabolic(np.array(range(len(T_ideal))), *fit)[i]
        T_desv_corr[i]=T_desv_mean[i]-pol4(np.array(range(len(T_ideal))), *fit1)[i]
        sigma_corr[i]=np.std(T_desv_corr[i])
        k_corr[i]=scipy.stats.kurtosis(T_desv_corr[i], fisher=True)
        skew_corr[i]=scipy.stats.skew(T_desv_corr[i])

    
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
    ax1.set_ylabel('T (K)', rotation=0, fontsize=16)
    ax1.yaxis.set_label_coords(0., 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.hlines(0, 0, 365, linewidth=0.6, linestyle='--')
    plt.yticks(fontsize=15)
    plt.xlim(0,370)
    plt.vlines(cut1, ymin=np.min(T_desv_mean), ymax=np.max(T_desv_mean), linestyle='dashed')
    plt.vlines(cut2, ymin=np.min(T_desv_mean), ymax=np.max(T_desv_mean), linestyle='dashed')
    plt.legend(loc='best', ncol=3)
    
    ax2=plt.subplot(412)
    plt.plot(np.arange(0,len(k),1)/4., sigma, label='standard dev', color='orange')
    plt.xticks(ticks=np.array(range(0,len(k),200)), fontsize=15)
    ax2.set_title('Standard deviation', fontsize=16)
    ax2.set_xlabel('DOY', fontsize=16)
    ax2.xaxis.set_label_coords(1., -0.2)
    ax2.set_ylabel('T (K)', rotation=0, fontsize=16)
    ax2.yaxis.set_label_coords(0., 1.1)
    ax2.set_xticks(np.array([0, 50, 100, 150, 200, 250, 300, 350]))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.hlines(0, 0, 365, linewidth=0.6, linestyle='--')
    plt.vlines(cut1, ymin=np.min(sigma), ymax=np.max(sigma), linestyle='dashed')
    plt.vlines(cut2, ymin=np.min(sigma), ymax=np.max(sigma), linestyle='dashed')
    plt.ylim(0, 15)
    plt.yticks(fontsize=15)
    plt.xlim(0,370)
    
    ax3=plt.subplot(413)
    plt.plot(np.arange(0,len(k),1)/4., skew, label='skewness', color='green')
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
    plt.vlines(cut1, ymin=np.min(k), ymax=np.max(k), linestyle='dashed')
    plt.vlines(cut2, ymin=np.min(k), ymax=np.max(k), linestyle='dashed')
    plt.xlim(0,370)
    
    ax4=plt.subplot(414)
    plt.plot(np.arange(0,len(k),1)/4., k, label='kutosis', color='red')
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
    plt.vlines(cut1, ymin=np.min(skew), ymax=np.max(skew), linestyle='dashed')
    plt.vlines(cut2, ymin=np.min(skew), ymax=np.max(skew), linestyle='dashed')
    plt.xlim(0,370)
    
    return T_desv_mean, sigma, k, skew

def corr(T_desv_mean, sigma, k, skew, year, cut1, cut2):
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
    plot

    '''
    plt.figure(7)
    plt.suptitle('Correlations with mean (%a)' %year , fontsize=24)
    ax1=plt.subplot(121)
    plt.xlabel('Deviations (mean) (T)', fontsize=24)
    plt.plot(T_desv_mean, k, '.', label='kurtosis')
    plt.plot(T_desv_mean[4*cut1:4*cut2], k[4*cut1:4*cut2], 'v', color='red', markersize=8)
    plt.ylabel('Kurtosis', fontsize=24)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylim(-3.5, 3.5)
    plt.xlim(-13,13)
    plt.hlines(0, -13, 13, linestyle='dashed', linewidth=0.5)
    plt.vlines(0,-3.5,3.5, linestyle='dashed', linewidth=0.5)
    
    ax2=plt.subplot(122)
    plt.xlabel('Deviations (mean) (T)', fontsize=24)
    plt.plot(T_desv_mean, skew, '.', label='skewness')
    plt.plot(T_desv_mean[4*cut1:4*cut2], skew[4*cut1:4*cut2], 'v', color='red', markersize=8)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Skewness', fontsize=24)
    plt.ylim(-2.5, 2.5)
    plt.xlim(-13,13)
    plt.hlines(0, -13, 13, linestyle='dashed', linewidth=0.5)
    plt.vlines(0,-2.5,2.5, linestyle='dashed', linewidth=0.5)
    
    plt.figure(8)
    plt.suptitle('Correlation with standard deviation (%a)' %year , fontsize=24)
    ax1=plt.subplot(121)
    plt.xlabel('Standard deviation (T)', fontsize=24)
    plt.plot(sigma, k, '.', label='kurtosis')
    plt.plot(sigma[4*cut1:4*cut2], k[4*cut1:4*cut2], 'v', color='red', markersize=8)
    plt.ylabel('Kurtosis', fontsize=24)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylim(-3.5, 3.5)
    plt.xlim(0,20)
    plt.hlines(0, 0, 20, linestyle='dashed', linewidth=0.5)
    plt.vlines(0,-3.5,3.5, linestyle='dashed', linewidth=0.5)

    ax2=plt.subplot(122)
    plt.plot(sigma, skew, '.', label='skewness')
    plt.plot(sigma[4*cut1:4*cut2], skew[4*cut1:4*cut2], 'v', color='red', markersize=8)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Skewness', fontsize=24)
    plt.ylim(-2.5, 2.5)
    plt.xlim(0,20)
    plt.hlines(0, 0, 20, linestyle='dashed', linewidth=0.5)
    plt.vlines(0,-2.5,2.5, linestyle='dashed', linewidth=0.5)
    
    plt.figure(10)
    plt.title('Correlation with skewness (%a)' %year , fontsize=24)
    plt.xlabel('Skewness ', fontsize=24)
    plt.plot(skew, k, '.', label='kurtosis')
    plt.plot(skew[4*cut1:4*cut2], k[4*cut1:4*cut2], 'v', color='red', markersize=8)
    plt.ylabel('Kurtosis', fontsize=24)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylim(-3.5, 3.5)
    plt.xlim(-2.5,2.5)
    plt.hlines(0, -2.5, 2.5, linestyle='dashed', linewidth=0.5)
    plt.vlines(0,-3.5,3.5, linestyle='dashed', linewidth=0.5)    
    return

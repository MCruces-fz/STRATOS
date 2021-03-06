#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:19:34 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)
"""
import numpy as np
import matplotlib.pyplot as plt

def PC_evolution(T9, direction, year, cut1, cut2):
    '''
    Parameters
    ----------
    T9 : array
        temperature data corresponding to the 9 levels of interest
        
    direction : array
        linear decomposition of PCs in terms of the 9 levels

    year : int
        year corresponding to data

    cut1 : int
        DOY corresponding to first cut

    cut2 : int
        DOY corresponding to second cut

    Returns
    -------
    PCs : array
        evolution of 4 main PCs during the year selected
        values correspond to the associated temperature
        
        the plot is also returned
    '''
    
    dir9=np.hstack((direction[:,0].reshape(-1,1),direction[:,2].reshape(-1,1),direction[:,6].reshape(-1,1),direction[:,8].reshape(-1,1),direction[:,11].reshape(-1,1),direction[:,15].reshape(-1,1),direction[:,17].reshape(-1,1), direction[:,23].reshape(-1,1), direction[:,36].reshape(-1,1)))

    PCs=np.zeros((len(T9), 4))
    for i in range(len(T9)):
        for j in range(4):
            PCs[i,j]=np.sum(T9[i]*dir9[j])/np.sum(dir9[j])
        
    fig=plt.figure(13)
    plt.suptitle(year, fontsize=24)
    fig.subplots_adjust (hspace=0.9)
    ax1=plt.subplot(411)
    ax1.plot(np.arange(len(T9)), PCs[:,0], label='PC1') 
    ax1.set_title('PC1', fontsize=16, color='tab:blue')
    ax1.set_xlabel('DOY', fontsize=16)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax1.xaxis.set_label_coords(1., -0.2)
    ax1.set_ylabel('T (K)', rotation=0, fontsize=16)
    ax1.yaxis.set_label_coords(0., 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,0]), 0, len(T9), linewidth=0.5, linestyle='--')
    plt.yticks(fontsize=15)
    plt.vlines(cut1*4, np.min(PCs[:,0]), np.max(PCs[:,0]), linestyle='--')
    plt.vlines(cut2*4, np.min(PCs[:,0]), np.max(PCs[:,0]), linestyle='--')
    plt.xlim(0,len(T9))
    
    ax2=plt.subplot(412)
    plt.plot(np.arange(len(T9)), PCs[:,1], label='PC2', color='orange')
    ax2.set_title('PC2', fontsize=16, color='orange')
    ax2.set_xlabel('DOY', fontsize=16)
    ax2.xaxis.set_label_coords(1., -0.2)
    ax2.yaxis.set_label_coords(0., 1.1)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,1]), 0, len(T9), linewidth=0.5, linestyle='--')
    plt.yticks(fontsize=15)
    plt.vlines(cut1*4, np.min(PCs[:,1]), np.max(PCs[:,1]), linestyle='--')
    plt.vlines(cut2*4, np.min(PCs[:,1]), np.max(PCs[:,1]), linestyle='--')
    plt.xlim(0,len(T9))
    
    ax3=plt.subplot(413)
    plt.plot(np.arange(len(T9)), PCs[:,2], label='PC3', color='green')
    plt.yticks(fontsize=15)
    ax3.set_title('PC3', fontsize=16, color='green')
    ax3.set_xlabel('DOY', fontsize=16)
    ax3.xaxis.set_label_coords(1., -0.2)
    ax3.yaxis.set_label_coords(0., 1.1)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,2]), 0, len(T9), linewidth=0.5, linestyle='--')
    plt.vlines(cut1*4, np.min(PCs[:,2]), np.max(PCs[:,2]), linestyle='--')
    plt.vlines(cut2*4, np.min(PCs[:,2]), np.max(PCs[:,2]), linestyle='--')
    plt.xlim(0,len(T9))
    
    ax4=plt.subplot(414)
    plt.plot(np.arange(len(T9)), PCs[:,3], label='PC4', color='red')
    ax4.set_title('PC4', fontsize=16, color='red')
    ax4.set_xlabel('DOY', fontsize=16)
    ax4.xaxis.set_label_coords(1., -0.2)
    plt.yticks(fontsize=15)
    ax4.yaxis.set_label_coords(0., 1.1)
    ax4.spines['top'].set_visible(False)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    ax4.spines['right'].set_visible(False)
    plt.hlines(np.mean(PCs[:,3]), 0, len(T9), linewidth=0.5, linestyle='--')
    plt.vlines(cut1*4, np.min(PCs[:,3]), np.max(PCs[:,3]), linestyle='--')
    plt.vlines(cut2*4, np.min(PCs[:,3]), np.max(PCs[:,3]), linestyle='--')
    plt.xlim(0,len(T9))
    return PCs

def PC_evol_ideal(T9, T9_id, direction_id, year, cut1, cut2):
    '''
    

    Parameters
    ----------
    T9 : array
        temperature data corresponding to the 9 levels of interest
        
    T9_id : array
        ideal temperature data corresponding to the 9 levels of interest
        
    direction_id 
        linear decomposition of ideal PCs in terms of the 9 levels
        
    year : int
        year corresponding to data

    cut1 : int
        DOY corresponding to first cut

    cut2 : int
        DOY corresponding to second cut

    Returns
    -------
    PCs_id : array
        same as before but calculated with ideal T instead of real data

    '''
    dir9_id=np.hstack((direction_id[:,0].reshape(-1,1),direction_id[:,2].reshape(-1,1),direction_id[:,6].reshape(-1,1),direction_id[:,8].reshape(-1,1),direction_id[:,11].reshape(-1,1),direction_id[:,15].reshape(-1,1),direction_id[:,17].reshape(-1,1), direction_id[:,23].reshape(-1,1), direction_id[:,36].reshape(-1,1)))
    
    for i in range(9):
        T9_id[:,i]=T9_id[:,i]+np.median(T9[:,i])
    
    PCid1=np.zeros((len(T9_id)))
    for i in range(len(T9_id)):
        for j in range(9):
            PCid1[i]=sum(dir9_id[0]*T9_id[i])/sum(dir9_id[0])
            
    PCid2=np.zeros((len(T9_id)))
    for i in range(len(T9_id)):
        for j in range(9):
            PCid2[i]= sum(dir9_id[1]*T9_id[i])/sum(dir9_id[1])
            
    PCs_id=np.zeros((len(T9),2))
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
    
def id_vs_real_PC(T9, direction_id, PCs_id, year, cut1, cut2):
    '''
    

    Parameters
    ----------
    T9 : array
        temperature data corresponding to the 9 levels of interest
        
    direction_id 
        linear decomposition of ideal PCs in terms of the 9 levels
        
    PCs_id : array
        temperature evolution of each PC calculated with ideal T
        
    year : int
        year corresponding to data

    cut1 : int
        DOY corresponding to first cut

    cut2 : int
        DOY corresponding to second cut

    Returns
    -------
    PCs : TYPE
        evolution of each PC in terms of temperature (real)

    '''
    dir9_id=np.hstack((direction_id[:,0].reshape(-1,1),direction_id[:,2].reshape(-1,1),direction_id[:,6].reshape(-1,1),direction_id[:,8].reshape(-1,1),direction_id[:,11].reshape(-1,1),direction_id[:,15].reshape(-1,1),direction_id[:,17].reshape(-1,1), direction_id[:,23].reshape(-1,1), direction_id[:,36].reshape(-1,1)))
    
    PC1=np.zeros((len(T9)))
    for i in range(len(T9)):
        for j in range(9):
            PC1[i]=sum(dir9_id[0]*T9[i])/sum(dir9_id[0])
            
    PC2=np.zeros((len(T9)))
    for i in range(len(T9)):
        for j in range(9):
            PC2[i]= sum(dir9_id[1]*T9[i])/sum(dir9_id[1])
            
    PCs=np.zeros((len(T9),2))
    PCs[:,0]=PC1 ; PCs[:,1]=PC2

    fig=plt.figure(15)
    fig.subplots_adjust (hspace=0.7)
    
    ax1=plt.subplot(211)
    plt.yticks(fontsize=15)
    plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
    plt.xlim(0,len(PCs[:,0]))
    plt.xlabel('DOY', fontsize=15)
    ax1.xaxis.set_label_coords(1, -0.15)
    plt.ylabel('T (K)', rotation='horizontal', fontsize=15)
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
    plt.xlim(-5.5, 5.5)
    plt.ylim(-15, 15)
    plt.vlines(0, -15, 15, linestyle='dashed')
    plt.hlines(0, -5.5, 5.5, linestyle='dashed')
    
    return PCs

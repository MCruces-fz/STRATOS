# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:50:13 2020

@author: Martina Feijoo (martina.feijoo@rai.usc.es)
STRATOS
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

g=9.8
R=287.04
def calculate_pressure(P,z,T):
    fac=(g*z)/(R*T)
    y=P*np.exp(fac)
    return y

def interpoli(val,datax,datay): #data is an array
    count=0
    for i in range(len(datay)):
        if datay[i]>val:
            count+=1
    if count==37:
        # we cannot interpol
        return 0
    else:
        c1=count-1 ;c2=count
        y_inf=datay[c1]
        y_sup=datay[c2]
        x_inf=datax[c1]
        x_sup=datax[c2]
        y=x_inf+((val-y_inf)/(y_sup-y_inf))*(x_sup-x_inf)
        return y
        

year='2016'

# read data
altura = pd.read_csv(year+'/santiago_height_press_'+year+'.txt', header=None , delimiter='\s+')
temperatura = pd.read_csv(year+'/santiago_temp_press_'+year+'.txt', header=None , delimiter='\s+')

pres=altura.loc[0,:] #pres=pressure levels [hPa]
pres=np.array(pres) ; pres=pres[1:] ; pres=np.stack(pres).astype(float)

dias=altura.loc[:,0] ; dias=np.array(dias) ; dias=dias[1:]


find=[0.5,16.5] #height (km) where we want to compute T

matriz_dataframe=dias.reshape((len(dias),1))

for n in range(len(find)):
    val=find[n]
    interpol_pres=np.zeros(len(dias))
    interpol_temp=np.zeros(len(dias))
    for k in range(len(dias)):
        altis=altura.loc[k+1,:] ; altis=np.array(altis)[1:] ; altis=np.stack(altis).astype(float)
        tempis=temperatura.loc[k+1,:] ; tempis=np.array(tempis)[1:] ; tempis=np.stack(tempis).astype(float)
        
        
        p=interpoli(val*1000,pres,altis/9.8) 
        
        
        '''Para obter finalmetne a presión tomamos a altura  ao nivel 1000 hPa (último nivel) e calculamos, c
        oa fórmula barómetrica,a presión a esa altura:
        P=P0*exp(-cte*H/T) con H a diferenza de alturas e T a temperatura media na capa, que tomamos como a T
        ao nivel de presión de 1000 hPa (o último nivel)'''
        
        if p==0: 
            T=tempis[-1] ; z=altis[-1]/9.8 ; P=1000.
            p=calculate_pressure(P,z,T) #Devolve en hPa
        
        interpol_pres[k]=p 
        
        
        tt=np.interp(p,pres,tempis)
        interpol_temp[k]=tt
    
    matriz_dataframe=np.concatenate((matriz_dataframe,interpol_temp.reshape(len(dias),1)),axis=1)


valores=pd.DataFrame(matriz_dataframe,columns=['dias','Temp-500m','Temp-16500m'])
#valores.to_csv('interp_temperaturas'+year+'.csv',sep=';', index=False, header=
               #['dias','Temp-500m','Temp-16500m'])
h_100=altura[11]
h_100=np.array([h_100[1:]/g])[0] #alturas do nivel p=100hPa en metros
               

#--------------------------------------------------------------------
#                              DELTA T
#--------------------------------------------------------------------

plt.figure(1)
plt.suptitle(r'$ \Delta T[h]  (%a)$' %np.int(year), fontsize=24)
ax1=plt.subplot(211)
plt.plot(np.arange(len(dias)), valores['Temp-500m'], label='T[h=0.5km]')
plt.hlines(np.mean(valores['Temp-500m']), 0, len(dias), linestyle='--', linewidth=0.5, label='mean(T[h=0.5km])')
plt.legend(fontsize=16)
plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
plt.xlim(0, len(dias))
plt.yticks(fontsize=15)
plt.xlabel('DOY', fontsize=15)
plt.ylabel('T(K)', fontsize=15, rotation='horizontal')
ax1.yaxis.set_label_coords(0., 1.)

ax2=plt.subplot(212)
plt.plot(np.arange(len(dias)), valores['Temp-16500m'], label='T[h=16.5km])')
plt.hlines(np.mean(valores['Temp-16500m']), 0, len(dias), linestyle='--', linewidth=0.5, label='mean(T[h=16.5km])')
plt.legend(fontsize=16)
plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
plt.xlim(0, len(dias))
plt.yticks(fontsize=15)
plt.xlabel('DOY', fontsize=15)
plt.ylabel('T(K)', fontsize=15, rotation='horizontal')
ax2.yaxis.set_label_coords(0., 1.)

#--------------------------------------------------------------------
#                              DELTA H (ground, MMP)
#--------------------------------------------------------------------
plt.figure(2)
plt.suptitle(r'$ \Delta h [p=100hPa] (%a)$' %np.int(year), fontsize=24)
ax=plt.subplot(111)
plt.plot(np.arange(len(dias)), h_100, label='h[p=100hPa]')
plt.xticks((4*np.array([0, 50, 100, 150, 200, 250, 300, 350])), labels=np.array([0, 50, 100, 150, 200, 250, 300, 350]), fontsize=15)
plt.xlim(0, len(dias))
plt.yticks((1000*np.array([15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9])), labels=np.array([15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9]), fontsize=15)
plt.xlabel('DOY', fontsize=15)
plt.ylabel('h(km)', fontsize=15, rotation='horizontal')
ax.yaxis.set_label_coords(0., 1.)
plt.hlines(np.mean(h_100), 0, len(dias), linestyle='--', linewidth=0.5, label='mean(h[p=100hPa])')
plt.legend(fontsize=16)


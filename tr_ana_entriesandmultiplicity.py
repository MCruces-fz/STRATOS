#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Tue Dec  1 13:17:54 2020
author: Martina Feijoo Fontán

analysis of tragaldabas entries excluding borders
"""

import numpy                 as np
import matplotlib.pyplot     as plt
import scipy.optimize        as sco
from   matplotlib            import cm
from   matplotlib.colors     import LogNorm
from   scipy                 import ndimage
from   os.path               import join    as join_path


plt.close('all')

#%% DEFINING FUNCTIONS

def make_lines(eigvals, eigvecs, mean):
        """Make lines a length of std dev."""
        std = np.sqrt(eigvals)
        vec =  std * eigvecs/ np.hypot(*eigvecs)
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    
def map_marginal(j):
    ''' with this function we can plot the cellmap with its marginal distrib
    we just have the plane index j as input '''
    
    # definitions for the axes of the fig
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    spacing = 0.01
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left+0.1, bottom + height + spacing, width-0.2, 0.2]
    rect_histy = [left + width + spacing-0.1, bottom, 0.2, height]
    
    
    fig = plt.figure(figsize=(6, 6))
    plt.suptitle('Cell entries ' + str(label[j]) + 
                 '\n total ' + str(int(T_tot[j])), fontsize=20, x=0.1, y=0.8)
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=16)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False, labelsize=14)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False, labelsize=14)
    
    # the scatter/cmap plot:
    im=ax_scatter.imshow(T_sub[j], cmap='jet', norm=LogNorm(vmin=1, vmax=1000))
    ax_scatter.scatter(cm[j][1],cm[j][0],color='black')
    mean = np.array([mean_x[j], mean_y[j]])
    ax_scatter.plot(*make_lines(eigvals[0], eigvecs[:,0], mean), marker='o', color='peachpuff')
    ax_scatter.plot(*make_lines(eigvals[1], eigvecs[:,1], mean), marker='o', color='peachpuff')
    ax_scatter.plot(*make_lines(lenght_main[j], main_dir, mean), marker='o', color='blue')
    ax_scatter.set_xlim((-0.5, 9.5))
    ax_scatter.set_ylim((-.5, 7.5))
    ax_scatter.set_xticks(np.array([0,1,2,3,4,5,6,7,8,9]))
    ax_scatter.vlines(cm[j][1], -0.5, 7.5, linestyle= '--', linewidth=0.8)
    ax_scatter.hlines(cm[j][0], -0.5, 9.5, linestyle= '--', linewidth=0.8)
    cbaxes = fig.add_axes([0.1, 0.1, 0.02, 0.6])
    cbar=plt.colorbar(im, cax=cbaxes)
    cbar.ax.tick_params(labelsize=16) 
    
    # marginal plots:
    ax_histx.bar(np.arange(len(T_sub[j][0])), cum_x[:,j], color='peachpuff')
    ax_histy.barh(np.arange(len(T_sub[j])), cum_y[:,j], color='peachpuff')
    ax_histx.vlines(mean_x[j], 0, np.max(cum_x[:,j]), linestyle= '--', linewidth=0.8)
    ax_histy.hlines(mean_y[j], 0, np.max(cum_y[:,j]), linestyle= '--', linewidth=0.8)
    
    ax_histy.set_ylim(-0.5,7.5)
    ax_histy.set_xlabel('cumulative \n (norm)', fontsize=18)
    ax_histy.xaxis.set_label_coords(.7,-0.05)
    ax_histy.set_title(r'<y> = ' + str(np.round(mean_y[j],2)) + ', \n $\sigma_{y}$ = ' + str(np.round(std_y[j],3)) , fontsize=20)
    
    ax_histx.set_xlim(-0.5, 9.5)
    ax_histx.set_ylabel('cumulative \n (norm)', fontsize=18, rotation='horizontal')
    ax_histx.yaxis.set_label_coords(0., 1.1)
    ax_histx.set_title(r'<x> = ' + str(np.round(mean_x[j],2)) + ',  $\sigma_{x}$ = ' + str(np.round(std_x[j],3)) , fontsize=20)
    
    # plt.show()
    return 


def read_txt(file_name):
    InFile = open(file_name,'r+')
    lines=InFile.readlines()
    T1=lines[1].strip().split('\t')
    T3=lines[3].strip().split('\t')
    T4=lines[5].strip().split('\t')

    T1 = [np.float(ti) for ti in T1]
    T3 = [np.float(ti) for ti in T3]
    T4 = [np.float(ti) for ti in T4]
    return np.array(T1), np.array(T3), np.array(T4)


def linear_fit(x,a,b):
    return a*x+b


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()


def intertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov


def main_direction(eigvals,eigvecs):
    main_dir = eigvals[0]*eigvecs[:,0] + eigvals[1]*eigvecs[:,1]
    if main_dir[0] / main_dir[1] < 0:
        main_dir = eigvals[0]*eigvecs[:,0] - eigvals[1]*eigvecs[:,1]
    return main_dir

#%% LOADING DATA 

data_entries = np.loadtxt( 'tr20335001620.hld_cell_entries.dat' )
data_mul = 'tr20329083640.hld_plane_mult.dat'

# selecting points we want to fit in mul
cut1=4; cut2=11

x=np.array([0,1,2,3,4,5,6,7,8,9])
y=np.array([0,1,2,3,4,5,6,7])

label=['T1', 'T3', 'T4']

T1_mul, T3_mul, T4_mul = read_txt(data_mul)
T=[T1_mul, T3_mul, T4_mul]

multiplicity = np.arange(1, len(T1_mul)+1, 1)
multiplicity_4 = np.arange(1, len(T4_mul)+1, 1)
mul=[multiplicity, multiplicity, multiplicity_4]

# we divide data from different planes and delete borders
T1_entries = data_entries[0 :10, :,] [1:-1, 1:-1]
T3_entries = data_entries[10:20, :,] [1:-1, 1:-1]
T4_entries = data_entries[20:  , :,] [1:-1, 1:-1]
T_entries = [T1_entries, T3_entries, T4_entries]


#%% CALC

# substracting the lowest multiplicity to each T
T_sub = np.array([T_entries[i] +1 - np.min(T_entries[i]) for i in range(3)])

# computing total entries
T_tot = np.array([np.sum(T_entries[i]) for i in range(3)])

# returns index of CM (x,y)
cm=np.array([ndimage.measurements.center_of_mass(T_sub[i]) for i in range(3)])

cum_x=np.zeros((len(T1_entries[0]),3))
cum_y=np.zeros((len(T1_entries),3))

bins_x=np.array([0,1,2,3,4,5,6,7,8,9,10]) - 0.5
bins_y=np.array([0,1,2,3,4,5,6,7,8]) - 0.5
    
mean_x = np.zeros(3) ; mean_y = np.zeros(3)
std_x = np.zeros(3) ; std_y = np.zeros(3)

lenght_main = np.zeros(3) ; angle_45 = np.zeros(3)

for j in range(3):
    for i in range(len(cum_x)):
        cum_x[i,j] = np.sum(T_sub[j][:,i])
    
    for i in range(len(cum_y)):
        cum_y[i,j] = np.sum(T_sub[j][i])
     
    #normalizing
    cum_x[:,j] = cum_x[:,j] / np.sum(cum_x[:,j])
    cum_y[:,j] = cum_y[:,j] / np.sum(cum_y[:,j])
    
    
    # computing mean and std
    for k in range(len(cum_x)):
       mean_x[j] += cum_x[k,j] * ((bins_x[k] + bins_x[k+1]) / 2) 
       std_x[j] += cum_x[k,j]*(bins_x[k] - mean_x[j])**2
    std_x[j] = np.sqrt(std_x[j])
    
    for k in range(len(cum_y)):
       mean_y[j] += cum_y[k,j] * ((bins_y[k] + bins_y[k+1]) / 2)
       std_y[j] += cum_y[k,j]*(bins_y[k] - mean_y[j])**2
    std_y[j] = np.sqrt(std_y[j]) 
    
    # covariance, eigvals, main direction, etc
    xbar, ybar, cov = intertial_axis((T_entries[j]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = main_direction(eigvals, eigvecs)
    lenght_main[j] = np.sqrt(main_dir[0]**2 + main_dir[1]**2)
    angle = np.arctan(main_dir[1] / main_dir[0])
    angle_45[j] = angle - np.pi/4
    
    # PLOT (cell map and marginal distributions, function above)
    map_marginal(j)
    


#%% LET'S FIT  (multiplicity)

x_fit = multiplicity[cut1:cut2]
T_fit = [[T[i][cut1:cut2]] for i in range(3)]

# fitting itself
fit = [ sco.curve_fit(linear_fit, np.log10(x_fit), np.log10(T_fit[i][0]), p0= (-1., 10), sigma=np.sqrt(np.log10(T_fit[i][0]))) for i in range(3)]
popt = [fit[i][0] for i in range(3)]


# COMPUTING DIFFERENCES BETWEEN REAL NUMBER OF EVENTS AND LINEAR FIT
dif = [0,0,0]
for j in range(3):
    fitted = np.array(linear_fit(np.log10(multiplicity) , *popt[j]))
    dif[j] = [np.log10(T[j][i]) - fitted[i] if T[j][i]!=0 else 0 for i in range(len(T[j]))]

Nleft = [np.sum(dif[j][0:3]) for j in range(3)]
Nright = [np.sum(dif[j][10:100]) for j in range(3)]


#%% MULTIPLICITY PLOT 

col=['cornflowerblue', 'sandybrown', 'limegreen']
darkcol=['mediumblue', 'darkorange', 'green']
label_fit=['fit T1', 'fit T3', 'fit T4']

x_ext = np.linspace(0, 100, 120) 

plt.figure(4)
plt.suptitle('Linear fit and differences', fontsize=24)

ax1=plt.subplot(211)
#plt.xlabel('log(multiplicity)', fontsize=20)
plt.ylim(-0.1,5) ; plt.xlim(-0.1, 2.2)
plt.ylabel('log(events)', fontsize=20)
plt.xticks(fontsize=16) ; plt.yticks(fontsize=16)
[plt.plot(np.log10(mul[j]), np.log10(T[j]), '.', label=label[j], color=darkcol[j]) for j in range(3)]
[plt.plot(np.log10(x_ext), linear_fit(np.log10(x_ext), *popt[j]), '-', label = label_fit[j], color=col[j]) for j in range(3)]
plt.legend(markerscale=2, fontsize=20, ncol=2)

ax2=plt.subplot(212)
plt.xlabel('log(multiplicity)', fontsize=20)
plt.ylabel('difference', fontsize=20)
plt.xticks(fontsize=16) ; plt.yticks(fontsize=16)
plt.xlim(-0.1, 2.2)
plt.hlines(0, 0, np.max(np.log10(multiplicity)), linestyle='--', linewidth=0.8)
for j in range(3):
    for i in range(len(dif[j])):
        if dif[j][i] != 0: 
            plt.plot(np.log10(mul[j][i]), dif[j][i], '.', color=darkcol[j])


#%% WRITING OUTPUT

f = open( join_path( "output_data", "mul_" + str(data_mul[2:13]) + "_fit.txt" ), "w+")
f.write( '# yeardaytime Ndat CM_position lenght theta slope intercept Nleft Nright \n')

for i in range(3):
    f.write('#' + str(label[i]) + '\n')
    
    f.write(str(data_mul[2:13]) + ' ' + str(T_tot[i]) + ' ' \
            + str(cm[i]) + ' ' + str(lenght_main[i]) + ' ' + str(angle_45[i]) + ' ' \
            + str(popt[i][0]) +' '+ str(popt[i][1]) +' ' + str(Nleft[i]) +' '+ str(Nright[i]) + '\n')     
f.close()

f = open(join_path( 'output_data', 'tragaldabas_entries.txt'), 'w+')
f.write('# Analysis of cell entries and multiplicities saved here for the 3 plains T1, T3 and T4 \n \
         # by columns we have the following, each plane is indicated before the corresponding magnitudes: \n \
         # YYDOYHHMMSS (yeardaytime): for example  20335001620 means year 2020, day 335, time 00:16:20 \n \
         # TotHits: total hits for each layer, excluding border cells \n \
         # XCoG: x coordinate of centre of gravity for each layer \n \
         # YCoG: y coordinate of centre of gravity for each layer \n \
         # sXGoG: standard deviation (x axis) of centre of gravity \n \
         # sYGoG: standard deviation (y axis) of centre of gravity \n \
         # MDLen: lenght of the main direction of data, calculated with eigenvalues and eigenvectors \n \
         # MDm45: angle between main direction and x-axis \n \
         # MSlop: slope of the multiplicities linear fit  (double log scale) \n \
         # MIncept: intercept of the multiplicities linear fit  (double log scale) \n \
         # MexsL: excess, number of counts above linear fit (corresponding to multiplicities 1, 2 and 3) \n \
         # MexsR: deficit, number of counts below linear fit corresponding to multiplicities 10 to 100 \n')
         
cm_write = [[np.round(cm[i][0],2), np.round(cm[i][1],2)] for i in range(3)]
columns = '# YYDOYHHMMSS, T1TotHits, T1XCoG, T1YCoG, T1sXGoG, T1sYCoG, T1MDLen, T1MDm45, T1MSlop, T1MIncpt, T1MExsL, T1MExsR, ' \
                       'T3TotHits, T3XCoG, T3YCoG, T3sXGoG, T3sYCoG, T3MDLen, T3MDm45, T3MSlop, T3MIncpt, T3MExsL, T3MExsR, ' \
                       'T4TotHits, T4XCoG, T4YCoG, T4sXGoG, T4sYCoG, T4MDLen, T4MDm45, T4MSlop, T4MIncpt, T4MExsL, T4MExsR \n'
f.write('\n')
f.write(columns)
l = [str(T_tot[i]) + ' ' \
            + str(cm_write[i][0]) + ' ' + str(cm_write[i][1]) + ' ' + str(np.round(std_x[i], 2)) + ' ' + str(np.round(std_y[i],2)) + ' ' + str(np.round(lenght_main[i],2)) + ' ' + str(np.round(angle_45[i],2)) + ' ' \
            + str(np.round(popt[i][0], 2)) + ' '+ str(np.round(popt[i][1], 2)) + ' ' + str(np.round(Nleft[i], 2)) +' ' + str(np.round(Nright[i],2)) for i in range(3)]
    
f.write(str(data_mul[2:13]) + ' ' + str(l[0]) + ' ' + str(l[1]) + ' ' + str(l[2]) + ' ' + '\n')     
f.close()    


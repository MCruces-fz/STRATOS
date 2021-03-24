import numpy                 as np
import matplotlib.pyplot     as plt
import scipy.optimize        as sco
from   matplotlib            import cm
from   matplotlib.colors     import LogNorm
from   scipy                 import ndimage
from   os.path               import join           as join_path
from   tr_ana_constants      import label

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
    
    plt.show()
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

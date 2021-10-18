import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io 
import scipy.optimize as sopt
import numpy as np

def plot_gauss_points(gauss_points, title):
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    plt.figure(figsize=(5,4))
    plt.scatter(gauss_points[:,0], gauss_points[:,1], s=1)
    plt.title(title)
    plt.savefig(title)
    plt.show()
    
def plot_scaler_field_on_gauss_points(gauss_points, field, title, cmap='jet'):
    
    plt.figure(figsize=(5, 4))
    plt.scatter(gauss_points[:,0], gauss_points[:,1], s=1, c=field, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.savefig(title)
    plt.show()

def debug_activations(mean, std):
    fig, ax = plt.subplots(sharex=True, ncols=1, nrows=mean.shape[1])
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    for i in range(mean.shape[1]):
        ax[i].errorbar(range(mean.shape[0]), mean[:,i], yerr=std[:,i]/2, label='Layer {0}'.format(i+1), elinewidth=0.1, c=np.random.uniform(size=(3,)))
        ax[i].legend(loc='upper left', fontsize=8)
    ax[0].set_title('Distribution of activation values')
    ax[-1].set_xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('activation_distribution')
    plt.show()

def debug_weights(mean, std):
    fig, ax = plt.subplots(sharex=True, ncols=2, nrows=mean.shape[1]//2)
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    for i in range(0,mean.shape[1],2):
        ax[i//2, 0].errorbar(range(mean.shape[0]), mean[:,i], yerr=std[:,i]/2, label='Layer {0}'.format(i//2+1), elinewidth=0.1, c=np.random.uniform(size=(3,)))
        ax[i//2, 1].errorbar(range(mean.shape[0]), mean[:,i+1], yerr=std[:,i+1]/2, label='Layer {0}'.format(i//2+1), elinewidth=0.1, c=np.random.uniform(size=(3,)))
        ax[i//2, 0].legend(loc='upper left', fontsize=8)
        ax[i//2, 1].legend(loc='upper left', fontsize=8)
    ax[0,0].set_title('Distribution of weights', fontsize=8)
    ax[0,1].set_title('Distribution of bias', fontsize=8)
    ax[-1,0].set_xlabel('Epochs')
    ax[-1,1].set_xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('weight_distribution')
    plt.show()
    
def debug_gradients(mean, std):
    fig, ax = plt.subplots(sharex=True, ncols=2, nrows=mean.shape[1]//2)
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    for i in range(0,mean.shape[1],2):
        ax[i//2, 0].errorbar(range(mean.shape[0]), mean[:,i], yerr=std[:,i]/2, label='Layer {0}'.format(i//2+1), elinewidth=0.1, c=np.random.uniform(size=(3,)))
        ax[i//2, 1].errorbar(range(mean.shape[0]), mean[:,i+1], yerr=std[:,i+1]/2, label='Layer {0}'.format(i//2+1), elinewidth=0.1, c=np.random.uniform(size=(3,)))
        ax[i//2, 0].legend(loc='upper left', fontsize=8)
        ax[i//2, 1].legend(loc='upper left', fontsize=8)
    ax[0,0].set_title('Weight gradients', fontsize=8)
    ax[0,1].set_title('Bias gradients', fontsize=8)
    ax[-1,0].set_xlabel('Epochs')
    ax[-1,1].set_xlabel('Epochs')
    plt.tight_layout()
    plt.savefig('gradient_distribution')
    plt.show()
 
def plot_scaler_field(field, shape, title, conditions = []):
    for condition in conditions:
        field[condition] = np.nan
    field = field.reshape(shape)
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    plt.figure(figsize=(5,4))
    plt.imshow(field, cmap='jet', origin='lower')
    bar = plt.colorbar()
    bar.ax.ticklabel_format(style="sci", scilimits=(0,0))
    plt.title(title)
    plt.savefig(title)
    plt.show()    

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io 
import scipy.optimize as sopt

def plot_gauss_points(gauss_points, title):
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    plt.figure(figsize=(5,4))
    plt.scatter(gauss_points[:,0], gauss_points[:,1], s=1)
    plt.title(title)
    plt.show()
    
def plot_scaler_field_on_gauss_points(gauss_points, field, title, cmap='jet'):
    plt.rcParams.update({'font.family':'serif',
                         'font.sans-serif':'Times New Roman',
                         'font.size':12})
    plt.figure(figsize=(5, 4))
    plt.scatter(gauss_points[:,0], gauss_points[:,1], s=1, c=field, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show()
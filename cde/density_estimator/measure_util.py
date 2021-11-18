
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_conditional_hist(dataset, x_cond=[0, 1, 2], ylim=(-8, 8), resolution=100, mode='pdf', show=True, prefix='', numpyfig=False, holdonfig=None,fsize=[5,5],fdpi=300,fxlabel="x",fylabel="y"):
    """ Generates a histogram plot of the conditional distribution if y is 1-dimensional

        Args:
          xlim: 2-tuple specifying the x axis limits
          ylim: specifying the y axis limits
          resolution: integer specifying the resolution of plot
        """

    if holdonfig is None:
      fig = plt.figure(dpi=fdpi,figsize=fsize)
    else:
      fig = holdonfig

    ax = fig.gca()
    
    labels = []

    if isinstance(resolution,tuple):
      resolution_arr = resolution
    else:
      resolution_arr = np.full((len(x_cond)), resolution)

    for i in range(len(x_cond)):
      ndim_x = len(x_cond[i])

      bin_edges = np.linspace(ylim[0], ylim[1], num=(resolution_arr[i]+1))
      width = (ylim[1]-ylim[0])/resolution_arr[i]
      Y = bin_edges[1:]-(width/2)
      
      # calculate values of distribution
      if mode == "pdf":
        conditioned_ds = dataset[np.where(np.all(dataset[:,1:]==x_cond[i],axis=1))]
        conditioned_ds = conditioned_ds[:,0]

        Z, bin_edges = np.histogram(conditioned_ds, bins=bin_edges, density=True)
        
      elif mode == "cdf":
        Z = []
      elif mode == "joint_pdf":
        Z = []


      label = "x="+ str(x_cond[i])  if ndim_x > 1 else 'x=%.2f' % x_cond[i]
      labels.append(label)

      plt_out = ax.plot(Y, Z, linestyle=':', label=label, marker=".")

    

    if holdonfig is None:
      #plt.legend([prefix + label for label in labels], loc='upper right')
      plt.legend(loc='upper right')
      plt.xlabel(fxlabel)
      plt.ylabel(fylabel)
    else:
      ax.legend(loc='upper right')
    
    if show:
      plt.show()

    if numpyfig:
      fig.tight_layout(pad=0)
      fig.canvas.draw()
      numpy_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
      numpy_img = numpy_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      return numpy_img

    return fig

def measure_percentile(dataset,x_cond,p_perc):

    conditioned_ds = dataset[np.where(np.all(dataset[:,1:]==x_cond[0],axis=1))]
    conditioned_ds = conditioned_ds[:,0]
    measured_perc = np.percentile(conditioned_ds, p_perc)
    avg = np.average(conditioned_ds)
    return measured_perc,len(conditioned_ds),avg
    
def measure_percentile_allsame(dataset,p_perc):

    conditioned_ds = dataset[:,0]
    measured_perc = np.percentile(conditioned_ds, p_perc)
    avg = np.average(conditioned_ds)
    return measured_perc,len(conditioned_ds),avg

def measure_tail(dataset,x_cond,y):

    conditioned_ds = dataset[np.where(np.all(dataset[:,1:]==x_cond[0],axis=1))]
    conditioned_ds = conditioned_ds[:,0]
    occ = sum(i > y for i in conditioned_ds)
    return occ/len(conditioned_ds)

def measure_tail_allsame(dataset,y):

    conditioned_ds = dataset[:,0]
    occ = sum(i > y for i in conditioned_ds)
    return occ/len(conditioned_ds)


def init_tail_index_hill(dataset,x_cond,beta):

    conditioned_ds = dataset[np.where(np.all(dataset[:,1:]==x_cond[0],axis=1))]
    conditioned_ds = conditioned_ds[:,0]
    conditioned_ds.sort()
    sorted_ds = conditioned_ds[::-1]
    lnx = np.log(sorted_ds)
    k = math.floor(beta*len(lnx))
    if k==0 :
      return 0,0,0,0
    
    alpha = 1/((1/k)*(sum(lnx[0:k]-[lnx[k]]*k)))
    xk = sorted_ds[k]
    n = len(lnx)
    return alpha,xk,k,n

def estimate_tail_index_hill(alpha,xk,k,n,x):
    return (k/n)*((x/xk)**(-alpha))
    
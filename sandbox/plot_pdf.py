from FitGammaLikelihood import FitGammaLikelihood
import numpy as np

from glob import glob

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


from argparse import ArgumentParser

parser = ArgumentParser(description='show single telescope')
parser.add_argument('-i','--indir',   type=str, default="./pdf")
parser.add_argument('-t','--teltype', type=str, default="LST")
parser.add_argument('-r','--run', type=str, default="*")
args = parser.parse_args()
    
    

def get_subplot(x, y):
    a = fit.hits.data
    b = fit.norm.data
    # remove all axes before either of the two we look at
    for i in range(min(x,y)):
        a = a[1:-1,...].sum(axis=0)
        b = b[1:-1,...].sum(axis=0)
    # remove all axes between the two we look at
    for i in range(abs(x-y)-1):
        a = a[:,1:-1,...].sum(axis=1)
        b = b[:,1:-1,...].sum(axis=1)
    # remove all axes beyond either the two we look at
    for i in range(dim-max(x,y)-1):
        a = a[...,1:-1].sum(axis=-1)
        b = b[...,1:-1].sum(axis=-1)

    a = a.astype(np.float)
    b = b.astype(np.float)        
    # add a small number to every 0 to prevent division by 0
    b[b == 0] = 0.001
    # normalise: total number of PE by total number of hit pixels
    
    c = a / b
    
    # if all elements are zero or smaller, log-scaling will fail
    # add small number to prevent
    #if True not in c>0: c[...] = .01

    if x == y:  return c[1:-1]
    elif x < y: return c[1:-1,1:-1]
    else:       return c[1:-1,1:-1].T


filelist = glob("{}/{}_{}_raw.npz".format(args.indir,args.teltype,args.run))
if len(filelist) == 0:
    print("no files found: {}".format(args.indir))
    from sys import exit
    exit()

fit = FitGammaLikelihood([],[])

for i, filename in enumerate(filelist):
    if i == 0:
        fit.read_raw(filename)
    else:
        fit_t = FitGammaLikelihood([],[])
        fit_t.read_raw(filename)
        fit.hits.data += fit_t.hits.data
        fit.norm.data += fit_t.norm.data

dim = fit.hits.dimension

fig, axes = plt.subplots(dim,dim,figsize=(12, 8))
for i in range(dim):
    for j in range(dim):
        ax = axes[i][j]
        img = get_subplot(i,j)
        ax.set_xlabel("{} / {}".format(fit.hits.labels[j], fit.hits.bin_edges[j].unit))
        if i == j:
            ax.semilogy( fit.hits.bin_edges[i][:-1], img) 
        else:
            ax.set_ylabel("{} / {}".format(fit.hits.labels[i], fit.hits.bin_edges[i].unit))
            im = ax.pcolor(img, norm=LogNorm(vmin=1), cmap=cm.hot )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(im, cax=cax)

plt.tight_layout(pad=0.)
plt.show()
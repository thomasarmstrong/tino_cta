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
parser.add_argument('-i','--indir',   type=str, default=".")
parser.add_argument('-t','--teltype', type=str, default="LST")
args = parser.parse_args()
    
    

def get_subplot(x, y):
    axes = []

    if x == y:
        for ax in range(1,dim):
            if x < ax: axes.append(1)
            else:      axes.append(0)
        a = fit.hits.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2]).sum(axis=axes[3])
        b = fit.norm.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2]).sum(axis=axes[3])
        
        # add a small number to every 0 to prevent division by 0
        b[b == 0] = 0.001
        # normalise: total number of PE by total number of hit pixels
        c = a / b
        # now set those bins to zero in the resulting array
        c[b == 0] = 0.
        # get rid of under- and overflow bins
        #print(c)
        c = c[1:-1]
        # if the histogram is empty, log scale fails -> add a small number to every elment
        if True in c[c>0]:
            return c
        else: 
            return c + 0.01
    else:
        # remove all axes before either of the two we look at
        for i in range(min(x,y)):
            axes.append(0)
        # remove all axes between the two we look at
        for i in range(abs(x-y)-1):
            axes.append(1)
        # remove all axes beyond either the two we look at
        for i in range(dim-max(x,y)):
            axes.append(2)
        # project to two axes by summing over the unwanted ones
        a = fit.hits.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2])
        b = fit.norm.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2])

        
        # add a small number to every 0 to prevent division by 0
        b[b == 0] = 0.001
        # normalise: total number of PE by total number of hit pixels
        c = a / b
        # now set those bins to zero in the resulting array
        c[b == 0] = 0.
        # get rid of under- and overflow bins
        c = c[1:-1,1:-1]
        if x < y:
            return c
        else:
            return c.T

"""            
E     = fit.pdf.data.sum(axis=1).sum(axis=1).sum(axis=1).sum(axis=1)
d     = fit.pdf.data.sum(axis=0).sum(axis=1).sum(axis=1).sum(axis=1)
delta = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=1).sum(axis=1)
rho   = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=0).sum(axis=1)
gamma = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=0).sum(axis=0)


E_d     = fit.pdf.data.sum(axis=2).sum(axis=2).sum(axis=2)
E_delta = fit.pdf.data.sum(axis=1).sum(axis=2).sum(axis=2)
E_rho   = fit.pdf.data.sum(axis=1).sum(axis=1).sum(axis=2)
E_gamma = fit.pdf.data.sum(axis=1).sum(axis=1).sum(axis=1)

d_E     = fit.pdf.data.sum(axis=2).sum(axis=2).sum(axis=2).T()
d_delta = fit.pdf.data.sum(axis=0).sum(axis=2).sum(axis=2)
d_rho   = fit.pdf.data.sum(axis=0).sum(axis=1).sum(axis=2)
d_gamma = fit.pdf.data.sum(axis=0).sum(axis=1).sum(axis=1)

delta_E     = fit.pdf.data.sum(axis=1).sum(axis=2).sum(axis=2).T()
delta_d     = fit.pdf.data.sum(axis=0).sum(axis=1).sum(axis=2).T()
delta_rho   = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=2)
delta_gamma = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=1)

rho_E     = fit.pdf.data.sum(axis=1).sum(axis=1).sum(axis=2).T()
rho_d     = fit.pdf.data.sum(axis=0).sum(axis=1).sum(axis=2).T()
rho_delta = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=2).T()
rho_gamma = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=0)

gamma_E     = fit.pdf.data.sum(axis=1).sum(axis=1).sum(axis=1).T()
gamma_d     = fit.pdf.data.sum(axis=0).sum(axis=1).sum(axis=1).T()
gamma_delta = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=1).T()
gamma_rho   = fit.pdf.data.sum(axis=0).sum(axis=0).sum(axis=0).T()
"""



filelist = glob("{}/{}_*_raw.npz".format(args.indir,args.teltype))

fit = FitGammaLikelihood([],[])

for i, filename in enumerate(filelist):
    if i == 0:
        fit.read_raw(filename)
    else:
        fit_t = FitGammaLikelihood([],[])
        fit_t.read_raw(filename)
        fit.hits.data += fit_t.hits.data
        fit.norm.data += fit_t.norm.data
        
fit.normalise()
dim = fit.pdf.dimension

from astropy import units as u

fig, axes = plt.subplots(dim,dim,figsize=(12, 8))
for i in range(dim):
    for j in range(dim):
        ax = axes[i][j]
        img = get_subplot(i,j)
        ax.set_xlabel(fit.pdf.labels[j])
        if i != j:
            ax.set_ylabel(fit.pdf.labels[i])
            im = ax.pcolor(img, norm=LogNorm(vmin=1), cmap=cm.hot )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            ax.semilogy( fit.hits.bin_edges[i][:-1], img) 

plt.tight_layout(pad=0.)
plt.show()
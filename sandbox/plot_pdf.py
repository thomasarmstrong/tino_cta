from FitGammaLikelihood import FitGammaLikelihood
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def get_subplot(x, y):
    axes = []

    if x == y:
        for ax in range(1,dim):
            if x < ax: axes.append(1)
            else:      axes.append(0)
        a = fit.hits.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2]).sum(axis=axes[3])
        b = fit.norm.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2]).sum(axis=axes[3])
        
        b[b == 0] = 0.001
        c = a / b
        c[b == 0] = 0.
        c = c[1:-1]
        if True in c[c>0]:
            return c
        else: 
            return c + 0.01
    else:
        
        for i in range(min(x,y)):
            axes.append(0)
        for i in range(abs(x-y)-1):
            axes.append(1)
        for i in range(dim-max(x,y)):
            axes.append(2)

        a = fit.hits.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2])
        b = fit.norm.data.sum(axis=axes[0]).sum(axis=axes[1]).sum(axis=axes[2])
        b[b == 0] = 0.001
        c = a / b
        c[b == 0] = 0.
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





fit = FitGammaLikelihood([],[])
#fit.read_pdf("test.npz")
fit.read_raw("test_raw.npz")
fit.normalise()
dim = fit.pdf.dimension

from astropy import units as u

fig, axes = plt.subplots(dim,dim,figsize=(12, 8))
for i in range(dim):
    for j in range(dim):
        ax = axes[i][j]
        if i != j:
            ax.pcolormesh(get_subplot(i,j), cmap=cm.hot )
        else:
            ax.semilogy( fit.hits.bin_edges[i][:-1], get_subplot(i,j)) 

plt.show()
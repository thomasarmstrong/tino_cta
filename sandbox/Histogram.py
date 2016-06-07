import numpy as np
from scipy import ndimage
#from numba import jit

class nDHistogram:
    def __init__(self, bin_edges, labels=[""]):
        
        for label, edge in zip(labels, bin_edges):
            if len(edge) < 2:
                print("need at least two bin edges per dimension -- "+label)
                exit(-1)

        self.dimension = len(bin_edges)
        self.bin_edges = bin_edges
        self.labels    = labels
        
        self.data = np.zeros( [ len(x)+1 for x in bin_edges ] )

            
    def __str__(self):
        string = "bin edges:\n"
        for label, edge in zip(self.labels, self.bin_edges):
            string += label
            string += "\t"
            string += str(edge)
            string += "\n"
        return string
    
    def find_bin(self, value, axis):
        # np.digitize seems to ignore units... convert arg to unit of axis as workaround
        try:
            return np.digitize(value.to(axis.unit), axis)
        except AttributeError:
            return np.digitize(value, axis)
        
    def find_bins(self, args):
        bins = []
        for ij, (arg, axis) in enumerate( zip(args,self.bin_edges)):
            # np.digitize seems to ignore units... convert arg to unit of axis as workaround
            try:
                bins += [ np.digitize(arg.to(axis.unit), axis) ]
            except AttributeError:
                bins += [ np.digitize(arg, axis) ]
        return bins
    
    def fill(self, args, value=1):
        self.data[ self.find_bins(args) ] += value
    
    def evaluate(self, args):
        return self.data[ self.find_bins(args) ]
    

    def interpolate(self, args, out_of_bounds_value=0.,order=3):
        bins_u = np.array(self.find_bins(args))
        bins_l = bins_u - 1
        
        margs       = np.zeros(len(args))
        bin_u_edges = np.zeros(len(args))
        bin_l_edges = np.zeros(len(args))
        for ij, axis in enumerate(self.bin_edges):
            unit = args[ij].unit
            margs[ij]       = args[ ij ].value
            bin_u_edges[ij] = axis[ bins_u[ij]-1 ].to(unit).value
            bin_l_edges[ij] = axis[ bins_l[ij]-1 ].to(unit).value
        coordinates =  (margs-bin_u_edges) / (bin_u_edges-bin_l_edges) * (bins_u - bins_l) + bins_u
        
        mcoordinates = []
        for coor in coordinates:
            mcoordinates.append([coor])
        
        try:
            return ndimage.map_coordinates(self.data, mcoordinates, order=order)[0] * self.data.unit
        except AttributeError:
            return ndimage.map_coordinates(self.data, mcoordinates, order=order)[0]
        
    
    def fill_bin(self, value, args):
        self.data[ args ] += value
        
        
    def get_bin_content( self, args):
        data = self.data
        for arg in args:
            data = data[arg]
        return data
    
    def get_outlier(self):
        data = self.data
        while len(data.shape):
            data = data[1:-1].sum(axis=0)
        return self.data.sum() - data
    def get_overflow(self):
        data = self.data
        while len(data.shape):
            data = data[:-1].sum(axis=0)
        return self.data.sum() - data    
    def get_underflow(self):
        data = self.data
        while len(data.shape):
            data = data[1:].sum(axis=0)
        return self.data.sum() - data    

    
    def write(self, filename):
        np.savez_compressed(filename, data=self.data,
                                      axes=self.bin_edges,
                                      labels=self.labels)
        
    @classmethod
    def read(cls, filename):
        with np.load(filename) as data:
            histo = cls( data['axes'], data['labels'] )
            histo.data = data['data']
        return histo
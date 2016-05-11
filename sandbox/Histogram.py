from sys import exit
import numpy as np


class nDHistogram:
    def __init__(self, bin_edges, labels=[""]):
        
        for label, edge in zip(labels, bin_edges):
            if len(edge) < 2:
                print("need at least two bin edges per dimension -- "+label)
                exit(-1)

        self.dimension = len(bin_edges)
        self.bin_edges = bin_edges
        self.labels    = labels
        self.out_of_bound = 0.
        
        self.data = np.zeros( [ len(x)-1 for x in bin_edges ] )
        
    def __str__(self):
        string = "bin edges:\n"
        for label, edge in zip(self.labels, self.bin_edges):
            string += label
            string += "\t"
            string += str(edge)
            string += "\n"
        return string
    
    def find_bin(self, value, axis):
        for nbin, edge in enumerate(axis):
            if edge > value:
                return nbin-1
        return float('Inf')
    
    def fill_bin(self, value, args):
        if len(args) != self.dimension:
            print("inconsistent dimensions while filling histogram")
            exit(-1)
        self.data[ args ] += value
        
    def get_bin_content( self, args):
        if len(args) != self.dimension:
            print("inconsistent dimensions while filling histogram")
            exit(-1)
        data = self.data
        for arg in args:
            data = data[arg]
        return data
    
    def fill(self, value, args):
        if len(args) != self.dimension:
            print("inconsistent dimensions while filling histogram")
            exit(-1)
            
        bins = []
        for ij, arg in enumerate(args):
            bins += [ self.find_bin(arg, self.bin_edges[ij]) ]
        if -1 in bins or float('Inf') in bins:
            self.out_of_bound += value
            return
        
        self.data[ bins ] += value
        
        return 
    
    
    def write(self, filename):
        np.savez_compressed(filename, data=self.data,
                                      axes=self.bin_edges,
                                      labels=self.labels)
        
    @classmethod
    def read(cls, filename):
        with np.load(filename) as data:
            histo = nDHistogram( data['axes'], data['labels'] )
            histo.data = data['data']
        return histo
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
        
        """ 
        for some reason len(x)+1 does not work -- it crashes write function
        len(x+[0]) is fine though....
        """
        self.data = np.zeros( [ len(x+[0]) for x in bin_edges ] )
        
    def __str__(self):
        string = "bin edges:\n"
        for label, edge in zip(self.labels, self.bin_edges):
            string += label
            string += "\t"
            string += str(edge)
            string += "\n"
        return string
    
    def find_bin(self, value, axis):
        return np.digitize(value, axis)
        
    def find_bins(self, args):
        bins = ()
        for ij, (arg, axis) in enumerate( zip(args,self.bin_edges)):
            bins += ( np.digitize(arg, axis), )
        return bins
    
    def fill(self, value, args):
        self.data[ self.find_bins(args) ] += value
        return 1
    
    def evaluate(self, args):
        return self.data[ self.find_bins(args) ]
    
    def fill_bin(self, value, args):
        self.data[ args ] += value
        
        
    def get_bin_content( self, args):
        data = self.data
        for arg in args:
            data = data[arg]
        return data
    
    
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
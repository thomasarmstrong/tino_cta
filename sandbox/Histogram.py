from sys import exit
import numpy as np


class nDHistogram:
    def __init__(self, bin_edges, labels=[""]):
        self.dimension = len(bin_edges)
        self.bin_edges = bin_edges
        self.labels    = labels
        self.out_of_bound = 0.

        self.data = np.zeros( [ len(x) for x in bin_edges ] )
        
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
            print(nbin, edge)
            if edge > value:
                return nbin-1
        return float('Inf')
    
    def fill_bin(self, value, args):
        if len(args) != self.dimension:
            print("inconsistent dimensions while filling histogram")
            exit(-1)
        if self.dimension == 2:
            self.data[args[0]][args[1]] += value
            return
        if self.dimension == 5:
            self.data[args[0]][args[1]][args[2]][args[3]][args[4]] += value
            return


        data = self.data
        for arg in args:
            print(data)
            data = data[arg]
        print(data)
        data += value
        print(data)
        #print(self.data)
        
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
        print( bins )
        if -1 in bins or float('Inf') in bins:
            self.out_of_bound += value
            return
        self.fill_bin(value, bins)
        return 
    
    
        if self.dimension == 2:
            self.data[bins[0]] \
                     [bins[1]] += value
            return
        if self.dimension == 5:
            self.data[self.find_bin(args[0], self.bin_edges[0])] \
                     [self.find_bin(args[1], self.bin_edges[1])] \
                     [self.find_bin(args[2], self.bin_edges[2])] \
                     [self.find_bin(args[3], self.bin_edges[3])] \
                     [self.find_bin(args[4], self.bin_edges[4])] += value
            return
        
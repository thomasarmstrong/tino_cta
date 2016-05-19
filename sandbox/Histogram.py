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
        
        self.data = np.zeros( [ len(x)+1 for x in bin_edges ] )
        
        
        self.permutations = [ ]
        for i in range(2**self.dimension):
            seq = [0]*self.dimension
            for j, t in enumerate(seq):
                seq[j] = 1 if i & 2**j else 0
            self.permutations.append( seq )
        
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
        unit = axis.unit
        return np.digitize(value.to(unit), axis)
        
    def find_bins(self, args):
        bins = ()
        for ij, (arg, axis) in enumerate( zip(args,self.bin_edges)):
            # np.digitize seems to ignore units... convert arg to unit of axis as workaround
            unit = axis.unit
            bins += ( np.digitize(arg.to(unit), axis), )
        return bins
    
    def fill(self, value, args):
        self.data[ self.find_bins(args) ] += value
        return 1
    
    def evaluate(self, args):
        return self.data[ self.find_bins(args) ]
    
    def interpolate_linear(self, args):
        # TODO safeguard against edge querries
        bin_centres = []
        bins = []
        for axis, arg in zip(self.bin_edges,args):
            bin = self.find_bin(arg, axis)
            bins.append(bin)
            bin_centre = (axis[bin] + axis[bin-1])/2.
            bin_centres_t = [ (bin_centre,0) ]
            if arg > bin_centre:
                bin_centres_t.append( ((axis[bin] + axis[bin+1])/2.,1) )
            else:
                bin_centres_t.append( ((axis[bin-2] + axis[bin-1])/2.,-1) )
            bin_centres.append(bin_centres_t)
        
        result = 0.
        for seq in self.permutations:
            result_temp = 1
            data = self.data
            for digit, centre, bin, arg in zip(seq, bin_centres,bins,args):
                result_temp *= abs((arg-centre[digit^1][0])/(centre[1][0]-centre[0][0]))
                data = data[(bin)+centre[digit][1]]
            result += result_temp * data
        return result
    
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
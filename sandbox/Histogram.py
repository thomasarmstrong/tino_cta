import numpy as np

from numba import jit

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
        
        """ prepare some arrays needed for the interpolation """
        self.bins          = [0]*self.dimension
        self.bin_centres   = [0]*self.dimension
        self.bin_centres_t = [0]*2
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
        try:
            return np.digitize(value.to(axis.unit), axis)
        except AttributeError:
            return np.digitize(value, axis)
        
    def find_bins(self, args):
        bins = ()
        for ij, (arg, axis) in enumerate( zip(args,self.bin_edges)):
            # np.digitize seems to ignore units... convert arg to unit of axis as workaround
            try:
                bins += ( np.digitize(arg.to(axis.unit), axis), )
            except AttributeError:
                bins += ( np.digitize(arg, axis), )
        return bins
    
    def fill(self, args, value=1):
        self.data[ self.find_bins(args) ] += value
        return 1
    
    def evaluate(self, args):
        return self.data[ self.find_bins(args) ]
    
    @jit
    def interpolate_linear(self, args, out_of_bounds_value=0.):
        # TODO safeguard against edge querries (see next comment)
        for ii, (axis, arg) in enumerate(zip(self.bin_edges,args)):
            bin_ = self.find_bin(arg, axis)
            self.bins[ii] = bin_
            # for now, ignore pixel in under-/overflow bins; return default value
            if len(axis) > 2:
                if bin_ >= len(axis)-1 or bin_ <= 1: return out_of_bounds_value
            
            bin_centre = (axis[bin_] + axis[bin_-1])/2.
            self.bin_centres_t[0] = (bin_centre,0)

            if arg > bin_centre:
                self.bin_centres_t[1] = ((axis[bin_] + axis[bin_+1])/2.,1)
            else:
                self.bin_centres_t[1] = ((axis[bin_-2] + axis[bin_-1])/2.,-1)
            
            
            if len(axis) == 2:
                self.bin_centres_t[1] = ((3*axis[0] - axis[1])/2.,0)


            self.bin_centres[ii] = self.bin_centres_t[:]
        
        result = 0.
        for seq in self.permutations:
            result_temp = 1
            data = self.data
            for digit, centre, bin_, arg in zip(seq, self.bin_centres,self.bins,args):
                data = data[bin_+centre[digit][1]]
                result_temp *= abs((arg-centre[digit^1][0])/(centre[1][0]-centre[0][0]))
            result += result_temp * data
        return result.value
    
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
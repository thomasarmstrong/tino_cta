#!/usr/bin/env python2

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac

import glob
import sys

def sliding_window(my_list, window_size, step_size=None):
    if step_size is None:
        step_size = window_size
    start = 0
    while start+step_size < len(my_list):
        yield my_list[start:start+step_size]
        start += step_size
    else:
        yield my_list[start:]


astri_filelist_gamma = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                            "-user-c-ciro.bigongiari-MiniArray9-Simtel-gamma.lfns")
astri_filelist_proton = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                             "-user-c-ciro.bigongiari-MiniArray9-Simtel-proton.lfns")

pilot_args = ' '.join(('fit_events_hillas.py',
                       '-m 50',
                       '--tail',
                       '--events_dir ./',
                       '--store',
                       '--indir ./ --infile_list run*.gz'))

input_sandbox = ['modules', 'helper_functions.py',
                 # python wrapper for the mr_filter wavelet cleaning
                 '/local/home/tmichael/software/jeremie_cta/'
                 'sap-cta-data-pipeline/datapipe/',
                 # executable for the wavelet cleaning
                 '/local/home/tmichael/software/ISAP/ISAP/cxx/sparse2d/bin/mr_filter',
                 # fitsIO library
                 '/local/home/tmichael/software/fitsio/cfitsio/libcfitsio.so.5.3.39',
                 # sets up the environment + script that is being run
                 'dirac_pilot.sh', pilot_args.split()[0]]

print "\nrunning as:"
print pilot_args
print "\nwith input_sandbox:"
print input_sandbox

for astri_filelist in [astri_filelist_gamma, astri_filelist_proton]:
    for run_filelist in sliding_window([l.strip() for l in astri_filelist], 10):


        dirac = Dirac()

        j = Job()
        j.setCPUTime(500)

        print "\nrunning on {} files:".format(len(run_filelist))
        print run_filelist
        j.setInputData(run_filelist)

        continue


        j.setExecutable('dirac_pilot.sh', pilot_args)
        j.setInputSandbox( input_sandbox )
        j.setName('hillas fit test')


        if "dry" in sys.argv:
            print "running dry -- not submitting"
            break



        print "submitting job"
        res = dirac.submit(j)
        print 'Submission Result: ', res['Value']

        if "test" in sys.argv:
            print "test run -- only submitting one job"
            break


    break
    # since there are two nested loops, need to break again
    if "test" in sys.argv or "dry" in sys.argv:
        break


print "all done -- exiting now"
exit()




#### FAQ ####

# specify allowed sites to send the job to
j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it', 'LCG.GRIF.fr', 'LCG.CYFRONET.pl', 'LCG.PRAGUE-CESNET.cz', 'LCG.Prague.cz', 'LCG.LAPP.fr'])

# to specify input GRID files independent of the site the job is send to
file1='LFN:/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/gamma/run1011.simtel.gz'
j.setInputSandbox([file1])

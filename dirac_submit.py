#!/usr/bin/env python2

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac

import glob

dirac = Dirac()

j = Job()
j.setCPUTime(500)

file1='/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/gamma/run1011.simtel.gz'
file2='/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/proton/run10011.simtel.gz'
j.setInputData([file1])

pilot_args = ' '.join(('fit_events_hillas.py',
                       '-m 50',
                       '--tail',
                       '--events_dir ./',
                       '--store',
                       '-i . -f',
                       file1.split('/')[-1]))

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

print "running as: ", pilot_args
print "with input_sandbox:", input_sandbox

j.setExecutable('dirac_pilot.sh', pilot_args)
j.setInputSandbox( input_sandbox )
j.setName('hillas fit test')

print "submitting job"
res = dirac.submit(j)
print 'Submission Result: ', res['Value']

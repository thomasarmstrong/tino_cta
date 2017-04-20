#!/usr/bin/env python2

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac

import glob
import sys
import re



def sliding_window(my_list, window_size, step_size=None):
    if step_size is None:
        step_size = window_size
    start = 0
    while start+window_size < len(my_list):
        yield my_list[start:start+window_size]
        start += step_size
    else:
        yield my_list[start:]


dirac = Dirac()

astri_filelist_gamma = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                            "-user-c-ciro.bigongiari-MiniArray9-Simtel-gamma.lfns")
astri_filelist_proton = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                             "-user-c-ciro.bigongiari-MiniArray9-Simtel-proton.lfns")

pilot_args = ' '.join((
                       'fit_events_hillas.py',
                       #'classify_and_reconstruct.py', '--classifier_dir ./',
                       '-m 50',
                       '--tail',
                       '--events_dir ./',
                       '--out_file {}',
                       '--store',
                       '--indir ./ --infile_list run*.gz'))

input_sandbox = ['modules', 'helper_functions.py',

                 # python wrapper for the mr_filter wavelet cleaning
                 #'/local/home/tmichael/software/jeremie_cta/'
                 #'sap-cta-data-pipeline/datapipe/',
                 # executable for the wavelet cleaning
                 #'/local/home/tmichael/software/ISAP/ISAP/cxx/sparse2d/bin/mr_filter',
                 # fitsIO library
                 #'/local/home/tmichael/software/fitsio/cfitsio/libcfitsio.so.5.3.39',
                 # sets up the environment + script that is being run
                 'dirac_pilot.sh', pilot_args.split()[0]]

output_filename_template = 'rec_events_{}_{}_{}.h5'
#output_filename_template = 'classified_events_{}_{}_{}.h5'


print ("\nrunning as:")
print (pilot_args)
print ("\nwith input_sandbox:")
print (input_sandbox)
print ("\nwith output_sandbox file:")
print (output_filename_template
            .format('{particle-type}', '{cleaning-mode}', '{run-token}'))

for astri_filelist in [astri_filelist_gamma, astri_filelist_proton]:
    for run_filelist in sliding_window([l.strip() for l in astri_filelist], 1):

        j = Job()
        j.setCPUTime(500)
        j.setName('hillas fit test')
        j.setInputSandbox(input_sandbox)

        print ("\nrunning on {} files:".format(len(run_filelist)))
        print (run_filelist)

        input_data = run_filelist \
        #+ ['LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/meta/classifier/'
           #'classifier_{}_-K-C1-m3-s2,2,3-n4_RandomForestClassifier.pkl'.format(
                                #"tail" if "tail" in pilot_args else "wave")]
        j.setInputData(input_data)


        output_filename = output_filename_template.format(
                    "gamma" if "gamma" in " ".join(run_filelist) else "proton",
                    "tail" if "tail" in pilot_args else "wave",
                    '-'.join([re.split('/|\.', run_filelist[+0])[-3],
                              re.split('/|\.', run_filelist[-1])[-3]]))
        print ("\nOutputSandbox: {}".format(output_filename))
        j.setOutputSandbox([output_filename])

        j.setExecutable('dirac_pilot.sh', pilot_args.format(output_filename))


        if "dry" in sys.argv:
            print ("\nrunning dry -- not submitting")
            break


        print ("\nsubmitting job")
        res = dirac.submit(j)
        print ('Submission Result: {}'.format(res['Value']))

        if "test" in sys.argv:
            print ("test run -- only submitting one job")
            break


    # since there are two nested loops, need to break again
    if "test" in sys.argv or "dry" in sys.argv:
        break


print ("all done -- exiting now")
exit()




#### FAQ ####

# specify allowed sites to send the job to
j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it', 'LCG.GRIF.fr', 'LCG.CYFRONET.pl', 'LCG.PRAGUE-CESNET.cz', 'LCG.Prague.cz', 'LCG.LAPP.fr'])

# to specify input GRID files independent of the site the job is send to
file1='LFN:/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/gamma/run1011.simtel.gz'
j.setInputSandbox([file1])

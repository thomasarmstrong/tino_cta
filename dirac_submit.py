#!/usr/bin/env python2

import glob
import sys
import re

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac



def sliding_window(my_list, window_size, step_size=None):
    if step_size is None:
        step_size = window_size
    start = 0
    while start+window_size < len(my_list):
        yield my_list[start:start+window_size]
        start += step_size
    else:
        yield my_list[start:]


# this thing submits all the jobs
dirac = Dirac()


#   ######  ######## ######## ######## ########  #### ##    ##  ######
#  ##    ##    ##    ##       ##       ##     ##  ##  ###   ## ##    ##
#  ##          ##    ##       ##       ##     ##  ##  ####  ## ##
#   ######     ##    ######   ######   ########   ##  ## ## ## ##   ####
#        ##    ##    ##       ##       ##   ##    ##  ##  #### ##    ##
#  ##    ##    ##    ##       ##       ##    ##   ##  ##   ### ##    ##
#   ######     ##    ######## ######## ##     ## #### ##    ##  ######
do_tailcuts = False
wavelet_args = "-K -C1 -m3 -s2,2,3 -n4"

pilot_args = ' '.join((
                       'classify_and_reconstruct.py',
                       '--classifier ./{classifier}',
                       '-m 50',
                       '--tail' if do_tailcuts else '--wave_dir ./',
                       '--out_file {out_file}',
                       '--store',
                       '--indir ./ --infile_list run*.gz'))


# files containing lists of the ASTRI files on the GRID
astri_filelist_gamma = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                            "-user-c-ciro.bigongiari-MiniArray9-Simtel-gamma.lfns")
astri_filelist_proton = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                             "-user-c-ciro.bigongiari-MiniArray9-Simtel-proton.lfns")


# the pickled classifier on the GRID
classifier_LFN = "LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/meta/classifier/"\
                 "{}/classifier_{}_{}_{}.pkl".format(
                         "ASTRI",
                         "tail" if do_tailcuts else "wave",
                         wavelet_args.replace(' ', '').replace(',', ''),
                         "RandomForestClassifier")


# define a template name for the file that's going to be written out.
# the placeholder braces are going to get set during the file-loop
output_filename_template = './classified_events_{}_{}_{}.h5'

# sets all the local files that are going to be uploaded with the job
# plus the pickled classifier (if the file name starts with `LFN:`, it's will be copied
# from the GRID itself)
input_sandbox = ['modules', 'helper_functions.py',

                 # python wrapper for the mr_filter wavelet cleaning
                 '/local/home/tmichael/software/jeremie_cta/'
                 'sap-cta-data-pipeline/datapipe/',

                 # executable for the wavelet cleaning
                 '/local/home/tmichael/software/ISAP/ISAP/cxx/sparse2d/bin/mr_filter',

                 # sets up the environment + script that is being run
                 'dirac_pilot.sh', pilot_args.split()[0],

                 # the pickled model for the event classifier
                 classifier_LFN,

                 'LFN:/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/'
                 'gamma/run1011.simtel.gz'
                 ]


print("\nrunning as:")
print(pilot_args)
print("\nwith input_sandbox:")
print(input_sandbox)
print("\nwith output_sandbox file:")
print(output_filename_template.format(
        '{particle-type}', '{cleaning-mode}', '{run-token}'))

for i, astri_filelist in enumerate([astri_filelist_gamma, astri_filelist_proton]):
    # proton files are smaller, can afford more files per run
    window_size = [100, 500][i]
    for run_filelist in sliding_window([l.strip() for l in astri_filelist], 1):

        j = Job()
        j.setCPUTime(500)
        j.setName('hillas fit test')
        j.setInputSandbox(input_sandbox)

        print("\nrunning on {} file{}:".format(len(run_filelist),
                                               "" if len(run_filelist) == 1 else "s"))
        print(run_filelist)

        # setting input
        # j.setInputData(run_filelist)

        # setting output
        output_filename = output_filename_template.format(
                    "gamma" if "gamma" in " ".join(run_filelist) else "proton",
                    "tail" if "tail" in pilot_args else "wave",
                    # this selects the `runxxx` part of the first and last file in the run
                    # list and joins them with a dash so that we get a nice identifier in
                    # the outpult file name
                    '-'.join([re.split('/|\.', run_filelist[+0])[-3],
                              re.split('/|\.', run_filelist[-1])[-3]]))
        print("\nOutputSandbox: {}".format(output_filename))
        j.setOutputSandbox([output_filename, "mr_filter"])

        # the `dirac_pilot.sh` is the executable. it sets up the environment and then
        # starts the script will all parameters given by `pilot_args`
        j.setExecutable('dirac_pilot.sh',
                        pilot_args.format(out_file=output_filename,
                                          classifier=classifier_LFN.split('/')[-1]))

        # check if we should somehow stop doing what we are doing
        if "dry" in sys.argv:
            print("\nrunning dry -- not submitting")
            break

        # this sends the job to the GRID and uploads all the files in the input sandbox in
        # the process
        print("\nsubmitting job")
        res = dirac.submit(j)
        print('Submission Result: {}'.format(res['Value']))

        if "test" in sys.argv:
            print("test run -- only submitting one job")
            break

    # since there are two nested loops, need to break again
    if "test" in sys.argv or "dry" in sys.argv:
        break


print("all done -- exiting now")
exit()

# ########    ###     #######
# ##         ## ##   ##     ##
# ##        ##   ##  ##     ##
# ######   ##     ## ##     ##
# ##       ######### ##  ## ##
# ##       ##     ## ##    ##
# ##       ##     ##  ##### ##

# specify allowed sites to send the job to
j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it',
                  'LCG.GRIF.fr', 'LCG.CYFRONET.pl', 'LCG.PRAGUE-CESNET.cz',
                  'LCG.Prague.cz', 'LCG.LAPP.fr'])

# to specify input GRID files independent of the site the job is send to
file1 = 'LFN:/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/'\
        'gamma/run1011.simtel.gz'
j.setInputSandbox([file1])

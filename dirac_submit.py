#!/usr/bin/env python2

import os
import sys
import glob
import re

from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac


def sliding_window(my_list, window_size, step_size=None, start=0):
    if step_size is None:
        step_size = window_size
    start = start
    while start+window_size < len(my_list):
        yield my_list[start:start+window_size]
        start += step_size
    else:
        yield my_list[start:]


# list of files on my GRID SE space
# not submitting jobs where we already have the output
GRID_filelist = open('vo.cta.in2p3.fr-user-t-tmichael.lfns').read()

# this thing submits all the jobs
dirac = Dirac()
# checking Dirac for already submitted jobs sometimes needs a date
jobs_date = None
# jobs_date = '2017-08-29'


#   ######  ######## ######## ######## ########  #### ##    ##  ######
#  ##    ##    ##    ##       ##       ##     ##  ##  ###   ## ##    ##
#  ##          ##    ##       ##       ##     ##  ##  ####  ## ##
#   ######     ##    ######   ######   ########   ##  ## ## ## ##   ####
#        ##    ##    ##       ##       ##   ##    ##  ##  #### ##    ##
#  ##    ##    ##    ##       ##       ##    ##   ##  ##   ### ##    ##
#   ######     ##    ######## ######## ##     ## #### ##    ##  ######
mode = "tail" if "tail" in sys.argv else "wave"
cam_id_list = ["ASTRICam", "FlashCam"]
wavelet_args = None

pilot_args = ' '.join((
        'classify_and_reconstruct.py',
        '--classifier ./{classifier}',
        '--regressor ./{regressor}',
        '--out_file {out_file}',
        '--store',
        '--indir ./ --infile_list *.simtel.gz',
        '--tail' if "tail" in sys.argv else '',
        ))

# files containing lists of the ASTRI files on the GRID
astri_filelist_gamma = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                            "-user-c-ciro.bigongiari-MiniArray9-Simtel-gamma.lfns")
astri_filelist_proton = open("/local/home/tmichael/Data/cta/ASTRI9/vo.cta.in2p3.fr"
                             "-user-c-ciro.bigongiari-MiniArray9-Simtel-proton.lfns")

# files containing lists of the Prod3b files on the GRID
prod3b_filelist_gamma = open("/local/home/tmichael/Data/cta/Prod3b/"
                             "Paranal_gamma_North_20deg_HB9_merged.list")
prod3b_filelist_proton = open("/local/home/tmichael/Data/cta/Prod3b/"
                              "Paranal_proton_North_20deg_HB9_merged.list")


# proton files are smaller, can afford more files per run -- at a ratio 11:3
# window_sizes = [3*5, 11*5]  # ASTRI
if mode == "wave":
    window_sizes = [15, 15]
else:
    window_sizes = [60, 60]
# I used the first few files to train the classifier and regressor -- skip them
# start_runs = [14, 100]  # ASTRI
start_runs = [30, 30]


# the pickled classifier and regressor on the GRID
model_path_template = \
    "LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/meta/ml_models/{}/{}_{}_{}_{}.pkl"
classifier_LFN = model_path_template.format(
                                            # "astri_mini",
                                            "prod3b/paranal",
                                            "classifier",
                                            mode,
                                            "{cam_id}",
                                            "RandomForestClassifier")
regressor_LFN = model_path_template.format(
                                           # "astri_mini",
                                           "prod3b/paranal",
                                           "regressor",
                                           mode,
                                           "{cam_id}",
                                           "RandomForestRegressor")


# define a template name for the file that's going to be written out.
# the placeholder braces are going to get set during the file-loop
output_filename_template = 'classified_events_{}_{}_{}.h5'
# output_path = "cta/astri_mini/"
output_path = "cta/prod3b/paranal/"

# sets all the local files that are going to be uploaded with the job plus the pickled
# classifier (if the file name starts with `LFN:`, it will be copied from the GRID itself)
input_sandbox = ['modules', 'helper_functions.py',

                 # python wrapper for the mr_filter wavelet cleaning
                 '/local/home/tmichael/software/jeremie_cta/'
                 'sap-cta-data-pipeline/datapipe/',

                 # sets up the environment + script that is being run
                 'dirac_pilot.sh', pilot_args.split()[0],

                 # the executable for the wavelet cleaning
                 'LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/bin/mr_filter/v3_1/mr_filter',
                 ]
for cam_id in cam_id_list:
    # the pickled model for the event classifier
    input_sandbox.append(classifier_LFN.format(cam_id=cam_id))
    # the pickled model for the energy regressor
    input_sandbox.append(regressor_LFN.format(cam_id=cam_id))

#

# ########  ######## ########   #######
# ##     ## ##       ##     ## ##     ##
# ##     ## ##       ##     ## ##     ##
# ########  ######   ##     ## ##     ##
# ##   ##   ##       ##     ## ##     ##
# ##    ##  ##       ##     ## ##     ##
# ##     ## ######## ########   #######

# set here the `run_token` of files you want to resubmit
# submits everything if empty
redo = dict([
    # enter run_tokens here as first element of a pair
])
# if resubmit is given as a command line argument, use dirac to find all jobs with
# "Failed" status and resubmit those
if "resubmit" in sys.argv:
    redo = []
    redo_ids = []
    for status in ["Failed", "Stalled"]:
        try:
            redo_ids += dirac.selectJobs(status=status, date=jobs_date,
                                         owner="tmichael")['Value']
        except KeyError:
            print("KeyError -- no {} jobs present?".format(status))

    n_jobs = len(redo_ids)
    if n_jobs == 0:
        print("no jobs found to resubmit ... exiting now")
        exit()

    print("getting names from {} failed/stalled jobs... please wait..."
          .format(n_jobs))
    for i, id in enumerate(redo_ids):
        if ((1000*i)/n_jobs) % 10 == 0:
            print("\r{} %".format((100*i)/n_jobs)),
        jobname = dirac.attributes(id)["Value"]["JobName"]
        if mode not in jobname:
            continue
        redo.append(jobname.split('.')[-1])
    else:
        print("\n... done")
        redo = dict(zip(redo, redo_ids))


# ########  ##     ## ##    ## ##    ## #### ##    ##  ######
# ##     ## ##     ## ###   ## ###   ##  ##  ###   ## ##    ##
# ##     ## ##     ## ####  ## ####  ##  ##  ####  ## ##
# ########  ##     ## ## ## ## ## ## ##  ##  ## ## ## ##   ####
# ##   ##   ##     ## ##  #### ##  ####  ##  ##  #### ##    ##
# ##    ##  ##     ## ##   ### ##   ###  ##  ##   ### ##    ##
# ##     ##  #######  ##    ## ##    ## #### ##    ##  ######

# get list of run_tokens that are currently running / waiting
running_ids = []
running_tokens = []
for status in ["Running", "Waiting"]:
    try:
        running_ids += dirac.selectJobs(status=status,  # date=jobs_date,
                                        owner="tmichael")['Value']
    except KeyError:
        pass

n_jobs = len(running_ids)
if n_jobs > 0:
    print("getting names from {} running/waiting jobs... please wait..."
          .format(n_jobs))
    for i, id in enumerate(running_ids):
        if ((1000*i)/n_jobs) % 10 == 0:
            print("\r{} %".format((100*i)/n_jobs)),
        jobname = dirac.attributes(id)["Value"]["JobName"]
        if mode not in jobname:
            continue
        running_tokens.append(jobname.split('.')[-1])
    else:
        print("\n... done")

# summary before submitting
print("\nrunning as:")
print(pilot_args)
print("\nwith input_sandbox:")
print(input_sandbox)
print("\nwith output file:")
print(output_filename_template.format(
        '{particle-type}', '{cleaning-mode}', '{run-token}'))
if len(redo):
    print("\nredoing these tokens:")
    print(redo.keys())

for i, astri_filelist in enumerate([prod3b_filelist_gamma, prod3b_filelist_proton]):
    window_size = window_sizes[i]
    start_run = start_runs[i]
    for run_filelist in sliding_window([l.strip() for l in astri_filelist],
                                       window_size, start=start_run):

        channel = "gamma" if "gamma" in " ".join(run_filelist) else "proton"

        # this selects the `runxxx` part of the first and last file in the run
        # list and joins them with a dash so that we get a nice identifier in
        # the outpult file name. if there is only one file in the list, use only that one
        # run_token = re.split('/|\.|_', run_filelist[+0])[-3]  # ASTRI
        run_token = re.split('_', run_filelist[+0])[3]

        if len(run_filelist) > 1:
            # run_token = '-'.join([run_token, re.split('/|\.', run_filelist[-1])[-3]])
            run_token = '-'.join([run_token, re.split('_', run_filelist[-1])[3]])

        # if is a resubmitting, kill the old process, otherwis it would get resubmitted
        # again and again
        # (if you are afraid to miss a run, update `GRID_filelist` and submit normally)
        if run_token in redo:
            this_job_id = redo[run_token]
            print("\ndeleting JobID {}".format(this_job_id))
            dirac.deleteJob(this_job_id)

        # setting output name
        output_filename = output_filename_template.format(channel, mode, run_token)

        # if some jobs failed and you want to resubmit, add their token to `redo`
        if len(redo) and run_token not in redo:
            continue

        print("-" * 20)
        print("\n")

        # if job already running / waiting, skip
        if run_token in running_tokens:
            print("{} still running".format(run_token))
            continue

        # if file already in GRID storage, skip
        # (you cannot overwrite it there, delete it and resubmit)
        if output_filename in GRID_filelist:
            print("{} already on GRID SE".format(run_token))
            continue

        print("\nrunning on {} file{}".format(len(run_filelist),
                                              "" if len(run_filelist) == 1 else "s"))
        # print(run_filelist)

        j = Job()
        # runtime in seconds times 8 (CPU normalisation factor)
        j.setCPUTime(10 * 3600 * 8)
        j.setName('reconstruct {}.{}.{}'.format(channel, mode, run_token))

        # bad sites
        j.setBannedSites([
                'LCG.IN2P3-CC.fr',  # jobs fail immediately after start
                'LCG.CAMK.pl',      # no miniconda (bad vo configuration?)
                'LCG.CETA.es'])

        j.setInputSandbox(input_sandbox +  # []
                          # adding the data files into the input sandbox instead of input
                          # data to not be bound to the very busy frascati site
                          [f.replace('/vo', 'LFN:/vo') for f in run_filelist]
                          )

        # setting input
        # j.setInputData(run_filelist)

        # print("\nOutputSandbox: {}".format(output_filename))
        # j.setOutputSandbox([output_filename])

        print("OutputData: {}{}".format(output_path, output_filename))
        j.setOutputData([output_filename], outputSE=None, outputPath=output_path)

        # the `dirac_pilot.sh` is the executable. it sets up the environment and then
        # starts the script with all parameters given by `pilot_args`
        j.setExecutable('dirac_pilot.sh',
                        pilot_args.format(out_file=output_filename,
                                          regressor=os.path.basename(regressor_LFN),
                                          classifier=os.path.basename(classifier_LFN)))

        # check if we should somehow stop doing what we are doing
        if "dry" in sys.argv:
            print("\nrunning dry -- not submitting")
            break

        # this sends the job to the GRID and uploads all the files into the input sandbox
        # in the process
        print("\nsubmitting job")
        res = dirac.submit(j)
        print('Submission Result: {}'.format(res['Value']))

        # break if this is only a test submission
        if "test" in sys.argv:
            print("test run -- only submitting one job")
            break

    # since there are two nested loops, need to break again
    if "dry" in sys.argv:
        break


print("\nall done -- exiting now")
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
                  'LCG.Prague.cz', 'LCG.LAPP.fr', 'LCG.CIEMAT.es', 'LCG.PIC.es'])

# to specify input GRID files independent of the site the job is send to
file1 = 'LFN:/vo.cta.in2p3.fr/user/c/ciro.bigongiari/MiniArray9/Simtel/'\
        'gamma/run1011.simtel.gz'
j.setInputSandbox([file1])

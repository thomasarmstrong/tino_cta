#!/usr/bin/env python2

from os.path import basename
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


# this thing submits all the jobs
dirac = Dirac()


# list of files on my GRID SE space
# not submitting jobs where we already have the output
GRID_filelist = open('vo.cta.in2p3.fr-user-t-tmichael.lfns').read()


#   ######  ######## ######## ######## ########  #### ##    ##  ######
#  ##    ##    ##    ##       ##       ##     ##  ##  ###   ## ##    ##
#  ##          ##    ##       ##       ##     ##  ##  ####  ## ##
#   ######     ##    ######   ######   ########   ##  ## ## ## ##   ####
#        ##    ##    ##       ##       ##   ##    ##  ##  #### ##    ##
#  ##    ##    ##    ##       ##       ##    ##   ##  ##   ### ##    ##
#   ######     ##    ######## ######## ##     ## #### ##    ##  ######
sys.argv.append("tail")
mode = "tail" if "tail" in sys.argv else "wave"
cam_id_list = ["ASTRICam", "FlashCam"]
wavelet_args = None  # setting this would overwrite the camera-specific defaults

pilot = 'dirac_pilot.sh'
execute = 'classify_and_reconstruct.py'
pilot_args = ' '.join([execute,
        '--classifier ./{classifier}',
        '--regressor ./{regressor}',
        '--out_file {out_file}',
        '--store',
        '--indir ./ --infile_list *.simtel.gz',
        '--tail' if "tail" in sys.argv else '',
        ])


# files containing lists of the Prod3b files on the GRID
prod3b_filelist_gamma = open("/local/home/tmichael/Data/cta/Prod3b/"
                             "Paranal_gamma_North_20deg_HB9_merged.list")
prod3b_filelist_proton = open("/local/home/tmichael/Data/cta/Prod3b/"
                              "Paranal_proton_North_20deg_HB9_merged.list")


if mode == "wave":
    window_sizes = [15, 15]
else:
    window_sizes = [20, 20]
# I used the first few files to train the classifier and regressor -- skip them
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
output_path = "cta/prod3b/paranal/"

# sets all the local files that are going to be uploaded with the job plus the pickled
# classifier (if the file name starts with `LFN:`, it will be copied from the GRID itself)
input_sandbox = ['modules', 'helper_functions.py', 'snippets/append_tables.py',

                 # python wrapper for the mr_filter wavelet cleaning
                 '/local/home/tmichael/software/jeremie_cta/'
                 'sap-cta-data-pipeline/datapipe/',

                 # sets up the environment + script that is being run
                 pilot, execute,

                 # the executable for the wavelet cleaning
                 'LFN:/vo.cta.in2p3.fr/user/t/tmichael/cta/bin/mr_filter/v3_1/mr_filter',
                 ]
for cam_id in cam_id_list:
    # the pickled model for the event classifier
    input_sandbox.append(classifier_LFN.format(cam_id=cam_id))
    # the pickled model for the energy regressor
    input_sandbox.append(regressor_LFN.format(cam_id=cam_id))

#

# ########  ##     ## ##    ## ##    ## #### ##    ##  ######
# ##     ## ##     ## ###   ## ###   ##  ##  ###   ## ##    ##
# ##     ## ##     ## ####  ## ####  ##  ##  ####  ## ##
# ########  ##     ## ## ## ## ## ## ##  ##  ## ## ## ##   ####
# ##   ##   ##     ## ##  #### ##  ####  ##  ##  #### ##    ##
# ##    ##  ##     ## ##   ### ##   ###  ##  ##   ### ##    ##
# ##     ##  #######  ##    ## ##    ## #### ##    ##  ######

# get jobs from today and yesterday...
import datetime as d
days = []
for i in range(2):  # how many days do you want to look back?
    days.append((d.date.today()-d.timedelta(days=i)).isoformat())

# get list of run_tokens that are currently running / waiting
running_ids = []
running_tokens = []
for status in ["Waiting", "Running"]:
    for day in days:
        try:
            running_ids += dirac.selectJobs(status=status,  date=day,
                                            owner="tmichael")['Value']
        except KeyError:
            pass

n_jobs = len(running_ids)
if n_jobs > 0:
    print("getting names from {} running/waiting jobs... please wait..."
          .format(n_jobs))
    for i, id in enumerate(running_ids):
        if ((100*i)/n_jobs) % 5 == 0:
            print("\r{} %".format(((20*i)/n_jobs)*5)),
        jobname = dirac.attributes(id)["Value"]["JobName"]
        if mode not in jobname:
            continue
        running_tokens.append(jobname.split('_')[-1])
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

for i, astri_filelist in enumerate([
        prod3b_filelist_gamma,
        prod3b_filelist_proton]):
    window_size = window_sizes[i]
    start_run = start_runs[i]
    for run_filelist in sliding_window([l.strip() for l in astri_filelist],
                                       window_size, start=start_run):

        channel = "gamma" if "gamma" in " ".join(run_filelist) else "proton"

        # this selects the `runxxx` part of the first and last file in the run
        # list and joins them with a dash so that we get a nice identifier in
        # the outpult file name. if there is only one file in the list, use only that one
        run_token = re.split('_', run_filelist[+0])[3]
        if len(run_filelist) > 1:
            run_token = '-'.join([run_token, re.split('_', run_filelist[-1])[3]])

        print("-" * 20)

        # setting output name
        output_filename = output_filename_template.format(channel, mode, run_token)

        # if job already running / waiting, skip
        if run_token in running_tokens:
            print("\n{} still running\n".format(run_token))
            continue

        # if file already in GRID storage, skip
        # (you cannot overwrite it there, delete it and resubmit)
        if output_filename in GRID_filelist:
            print("\n{} already on GRID SE\n".format(run_token))
            continue

        j = Job()
        # runtime in seconds times 8 (CPU normalisation factor)
        j.setCPUTime(6 * 3600 * 8)
        j.setName('_'.join([channel, mode, run_token]))
        j.setInputSandbox(input_sandbox)

        # bad sites
        j.setBannedSites([
            'LCG.IN2P3-CC.fr',  # jobs fail immediately after start
            'LCG.CAMK.pl',      # no miniconda (bad vo configuration?)
            'LCG.Prague.cz',    # no miniconda (bad vo configuration?)
            'LCG.PRAGUE-CESNET.cz',    # no miniconda (bad vo configuration?)
            'LCG.OBSPM.fr',
            'LCG.LAPP.fr',      # no miniconda (bad vo configuration?)
            # 'LCG.PIC.es',       #
            # 'LCG.M3PEC.fr',     #
            # 'LCG.CETA.es'
        ])

        # mr_filter loses its executable property by uploading it to the GRID SE; reset
        j.setExecutable('chmod', '+x mr_filter')

        for run_file in run_filelist:
            run_token = re.split('_', run_file)[3]
            output_filename_temp = \
                output_filename_template.format(channel, mode, run_token)

            # wait for a random number of seconds (up to five minutes) before starting
            # to add a bit more entropy in the starting times of the dirac querries
            sleep = random.randint(0, 5*60)
            j.setExecutable('sleep', sleep)

            # consecutively downloads the data files, processes them, deletes the input
            # and goes on to the next input file; afterwards, the output files are merged
            j.setExecutable('dirac-dms-get-file', "LFN:"+run_file)

            # the pilot script sets up the environment, i.e. ctapipe through miniconda
            # and then executes the `pilot_args` through python
            j.setExecutable(pilot,
                            pilot_args.format(out_file=output_filename_temp,
                                              regressor=basename(regressor_LFN),
                                              classifier=basename(classifier_LFN)))
            j.setExecutable('rm', basename(run_file))

        j.setExecutable('ls', '-lh')

        # pilot necessary again to run the python script
        j.setExecutable(pilot, ' '.join([
            'append_tables.py',
            '--events_dir ./',
            '--no_auto',
            '--out_file', output_filename
        ]))

        print("\nOutputData: {}{}".format(output_path, output_filename))
        j.setOutputData([output_filename], outputSE=None, outputPath=output_path)

        # check if we should somehow stop doing what we are doing
        if "dry" in sys.argv:
            print("\nrunning dry -- not submitting")
            break

        # this sends the job to the GRID and uploads all the
        # files into the input sandbox in the process
        print("\nsubmitting job")
        print('Submission Result: {}\n'.format(dirac.submit(j)['Value']))

        # break if this is only a test submission
        if "test" in sys.argv:
            print("test run -- only submitting one job")
            break

    # since there are two nested loops, need to break again
    if "dry" in sys.argv or "test" in sys.argv:
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

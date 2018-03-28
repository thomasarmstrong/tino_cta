#!/usr/bin/env python2

import sys
from os.path import basename, expandvars
from glob import glob

import argparse
import subprocess


from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Dirac import Dirac

dirac = Dirac()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--file_list', default=None,
                    help='lfns file list with files to download -- if not set, download automatically')
parser.add_argument('--match', default='', help="pattern that files in the list have to match to get downloaded")
parser.add_argument('--outdur', default='', help="destination for the downloaded files")

args = parser.parse_args()


# if no lfns file is given, generate one for the current user
if args.file_list is None:
    batcmd = 'dirac-dms-user-lfns'
    result = subprocess.check_output(batcmd, shell=True)
    file_list = result.split()[-1]
else:
    file_list = args.file_list

# try reading the lfns file
try:
    GRID_file_list = open(file_list).read()
except IOError:
    raise IOError("cannot read lfns file list...")


file_collection = []
for line in GRID_file_list:
    line = line.strip()

    if args.match not in line:
        continue

    # don't download if already in current directory
    if glob(basename(line)):
        continue

    if len(file_collection) < 100:
        file_collection.append(line)
    else:
        dirac.getFile(file_collection, destDir=args.outdir)
        file_collection = []

if file_collection:
    dirac.getFile(file_collection, destDir=args.outdir)

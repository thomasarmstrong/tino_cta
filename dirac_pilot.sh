#!/bin/bash

pwd
ls -lhrt
echo $PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH

export MINICONDA=/cvmfs/cta.in2p3.fr/software/miniconda
export PATH=$MINICONDA/bin:$PATH
source $MINICONDA/bin/activate ctapipe_0.3.4

echo $PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
which python

export MPLBACKEND=Agg

python $1

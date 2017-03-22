#!/bin/bash

export MINICONDA=/cvmfs/cta.in2p3.fr/software/miniconda
export PATH=$MINICONDA/bin:$PATH
source $MINICONDA/bin/activate ctapipe_0.3.4

python $1

#!/bin/bash

echo "pwd:"
pwd

echo "ls -lhrt"
ls -lhrt

export MINICONDA=/cvmfs/cta.in2p3.fr/software/miniconda
export PATH=$MINICONDA/bin:$PATH
source $MINICONDA/bin/activate ctapipe_v0.4

export PATH=./:$PATH
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH

echo '$PATH'
echo $PATH
echo '$LD_LIBRARY_PATH'
echo $LD_LIBRARY_PATH
echo '$PYTHONPATH'
echo $PYTHONPATH

echo 'which python'
which python

export MPLBACKEND=Agg

echo calling: 'python $@'
python $@


echo
echo "final ls -lhrt"
ls -lhrt

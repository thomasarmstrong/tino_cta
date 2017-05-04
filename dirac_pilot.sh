#!/bin/bash

echo "pwd:"
pwd

echo "ls -lhrt"
ls -lhrt

# setting some paths
export PATH=./:$PATH
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH

# sourcing the ctapipe miniconda environment
export MINICONDA=/cvmfs/cta.in2p3.fr/software/miniconda
export PATH=$MINICONDA/bin:$PATH
source $MINICONDA/bin/activate ctapipe_v0.4


echo '$PATH'
echo $PATH
echo '$LD_LIBRARY_PATH'
echo $LD_LIBRARY_PATH
echo '$PYTHONPATH'
echo $PYTHONPATH

echo 'which python'
which python

# prevent matplotlib to complain about missing backends
export MPLBACKEND=Agg

# getting and compiling cfitsio on site
{
    wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
    tar -xzvf cfitsio_latest.tar.gz
    cd cfitsio
    mkdir build
    ./configure --prefix=$PWD/build
    make
    make install
    export CFITSIO=$PWD/build
    export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
    cd ..
} # &> /dev/null

echo 'find . -name "libcfitsio*"'
find . -name "libcfitsio*"
# find . -name "libcfitsio*" -exec ln -s {} ./libcfitsio.so.5 \;

# ln -s ./cfitsio/build/lib/libcfitsio.a  libcfitsio.so.5
# ln -s ./cfitsio/build/lib/{libcfitsio.a,libcfitsio.so.5}

echo "linking and finding again"
find . -name "libcfitsio*"


# executing the script that we intend to run
echo calling: 'python $@'
python $@


echo
echo "final ls -lhrt"
ls -lhrt

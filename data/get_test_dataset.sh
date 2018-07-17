#!/bin/bash

URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz
TAR_FILE=./data/edges2shoes.tar.gz
wget $URL -O $TAR_FILE
tar -xvzf TAR_FILE

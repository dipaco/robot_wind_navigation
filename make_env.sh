#!/bin/bash

# Trick to be able to use conda on a shell script. Others use <conda init bash> instead
source ~/.bashrc

# Creates and activates conda environment
conda create -n $1 python=3.8 -y
conda activate $1
conda install pytorch==1.9.0 -c pytorch -y
conda install pytorch-geometric -c rusty1s -c conda-forge -y
conda install -c conda-forge fluidsim gym tensorboardx -y
conda install -c anaconda netcdf4 -y
conda install -c omnia termcolor -y
conda install -c conda-forge opencv -y


python -m pip install git+git://github.com/facebookresearch/hydra@0.11_branch --upgrade --force-reinstall
#python -m pip install git+git://github.com/denisyarats/dmc2gym.git --force-reinstall
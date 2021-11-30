#!/bin/bash
conda create --yes --name segnn python=3.8 numpy scipy matplotlib
source ~/anaconda3/etc/profile.d/conda.sh
conda activate segnn

conda install pytorch=1.9 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda install gcc_linux-64 -y
conda install pytorch-geometric -c rusty1s -c conda-forge -y
pip3 install e3nn
pip3 install rdkit-pypi
pip3 install wandb

#!/bin/bash

echo Startin Manylinux....
cd rational_activations
git pull origin master
./pypi_build_scripts/make_manylinux.sh
ls wheelhouse
python 3.6 -m pip install wheelhouse/cuda10.1/rational_activations-0.1.0-cp38-cp38-manylinux2014_x86_64.whl
python --version
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('Cuda available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version used by PyTorch:', torch.version.cuda)"
nvcc --version
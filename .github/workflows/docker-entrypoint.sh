#!/bin/bash

export CUDA_HOME=/usr/local/cuda-$(CUDA)/
export PATH=/usr/local/cuda-$(CUDA)/bin:$$PATH
echo CUDA_HOME: $$CUDA_HOME
nvcc --version
python3.7 -c "import sys; print('Python version:', sys.version)"
pip3.7 install airspeed pytest
pip3.7 install -r requirements.txt
python3.7 -c "import torch; print('CUDA version:', torch.version.cuda)"
python3.7  setup.py develop --user
python3.7 -m pytest


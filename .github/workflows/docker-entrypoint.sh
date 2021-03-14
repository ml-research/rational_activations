#!/bin/bash

echo CUDA_HOME: $$CUDA_HOME
pwd
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('CUDa version:', torch.version.cuda)"
pip install airspeed pytest
pip install -r requirements.txt
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python  setup.py develop --user
python -m pytest


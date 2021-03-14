#!/bin/bash

echo CUDA_HOME: $$CUDA_HOME
pwd
python3.7 -c "import sys; print('Python version:', sys.version)"
python3.7 -c "import torch; print('CUDa version:', torch.version.cuda)"
python3.7 -c "import cartopy; print('auasassss')"
pip3.7 install airspeed pytest
pip3.7 install -r requirements.txt
python3.7 -c "import torch; print('CUDA version:', torch.version.cuda)"
python3.7  setup.py develop --user
python3.7 -m pytest


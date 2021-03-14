#!/bin/bash

echo CUDA_HOME: $$CUDA_HOME
pwd
conda activate cicd_env_cuda10.1py3.7
python3.7 -c "import sys; print('Python version:', sys.version)"
python3.7 -c "import torch; print('CUDa version:', torch.version.cuda)"
python3.7 -c "import cartopy; print('auasassss')"
python3.7  setup.py develop --user
python3.7 -m pytest
#!/bin/bash

set -e

_V=0
while getopts "v" OPTION
do
  case $OPTION in
    v) _V=1
       ;;
  esac
done

CUDA_list=("cuda-10.1" "cuda-10.2" "cuda-11.0")

######### CHECKING ###############


if [ ! python3.7 -c "" &> /dev/null || ! python3.8 -c "" &> /dev/null ]; then
  printf "python3.7 and/or python3.8 not installed\n Installing...\n"
  bash pypi_build_scripts/install_all_python.sh
fi


python_list=(python3.6 python3.7 python3.8)
torch_lib_list=(
  "/usr/local/lib64/python3.6/site-packages/torch/lib/"
  "/usr/local/lib/python3.7/site-packages/torch/lib/"
  "/usr/local/lib/python3.8/site-packages/torch/lib/")

for j in 0 1 2
do
  CUDA_V=${CUDA_list[$j]}
  CUDA_LIB="/usr/local/$CUDA_V/lib64/"
  export CUDA_HOME="/usr/local/$CUDA_V"

  printf "Checking if python versions are correctly accessible and path to torch lib packages\n"
  for i in 0 1 2
  do
    PYTHON_V=${python_list[$i]}
    TORCH_LIB=${torch_lib_list[$i]}
    if [[ ! $TORCH_LIB =~ "$PYTHON_V" ]] || [ ! -d "$TORCH_LIB" ]
    then
      printf "Please provide a valid python torch lib path in \$TORCH_LIB\
              \ne.g. /usr/lib64/python3.6/site-packages/torch/lib/\
              \n$TORCH_LIB is not valid\n"
      exit 1
    fi

    if [ -z "$CUDA_LIB" ] || [ ! -d "$CUDA_LIB" ]
    then
      printf "Please provide a valid cuda lib path in \$CUDA_LIB\
              \ne.g. /usr/local/$CUDA_V/lib64/\n"
      exit 1
    fi
  done
  printf "All version correctly install with Rational's dependencies\n"
  unset PYTHON_V TORCH_LIB

  ######### CHECKING DONE  ###############


  # generate the wheels
  for i in 0 1 2
  do
    PYTHON_V=${python_list[$i]}
    TORCH_LIB=${torch_lib_list[$i]}
    export LD_LIBRARY_PATH=/usr/local/lib:$TORCH_LIB:$CUDA_LIB  # for it to be able to find the .so files
    rm -f /usr/local/cuda
    ln -s /usr/local/$CUDA_LIB /usr/local/cuda
    $PYTHON_V setup.py bdist_wheel
    set -- "${@:2}"
    source pypi_build_scripts/complete_wheel_repair.sh
    $PYTHON_V setup.py clean
  done
done
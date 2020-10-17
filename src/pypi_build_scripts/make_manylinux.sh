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

function log () {
    if [[ $_V -eq 1 ]]; then
        printf "$@\n"
    fi
}

# docker run -ti --gpus all --name manyl_cuda101 -v `pwd`:/prauper_src soumith/manylinux-cuda101:latest bash
# docker run -ti --gpus all --name manyl_cuda101 -v `pwd`:/prauper_src soumith/manylinux-cuda100:latest bash
if [ ! python3.7 -c "" &> /dev/null || ! python3.8 -c "" &> /dev/null ]; then
  printf "python3.7 and/or python3.8 not installed\n Installing...\n"
  bash install_all_python.sh
fi


python_list=(python3.6 python3.7 python3.8)
torch_lib_list=(
  "/usr/local/lib64/python3.6/site-packages/torch/lib/"
  "/usr/local/lib/python3.7/site-packages/torch/lib/"
  "/usr/local/lib/python3.8/site-packages/torch/lib/")
CUDA_LIB="/usr/local/cuda/lib64/"


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
    printf "Please provide a valid cyda torch lib path in \$CUDA_LIB\
            \ne.g. /usr/local/cuda/lib64/\n"
    exit 1
  fi
done
printf "All version correctly install with PAU's dependencies\n"
unset PYTHON_V TORCH_LIB


# generate the wheels
for i in 2
do
  PYTHON_V=${python_list[$i]}
  TORCH_LIB=${torch_lib_list[$i]}
  export LD_LIBRARY_PATH=/usr/local/lib:$TORCH_LIB  # for it to be able to find the .so files
  # $PYTHON_V setup.py bdist_wheel
  set -- "${@:2}"
  source ./complete_wheel_repair.sh
  $PYTHON_V setup.py clean
done

# use auditwheel to repair

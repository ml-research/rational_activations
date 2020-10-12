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

if [-z "$TORCH_LIB"] || [! -d "$TORCH_LIB"]
then
  printf "Please provide a valid python torch lib path in \$TORCH_LIB
          \n e.g. /usr/local/lib64/python3.6/site-packages/torch/lib/"
  exit 1
fi

if [-z "$CUDA_LIB"] || [! -d "$CUDA_LIB"]
then
  printf "Please provide a valid cyda torch lib path in \$CUDA_LIB
          \n e.g. /usr/local/cuda/lib64/"
  exit 1
fi

auditwheel -v repair --plat manylinux2014_x86_64 dist/pau-0.0.16-cp36-cp36m-linux_x86_64.whl
cd wheelhouse/
python3 -m wheel unpack pau-*.whl
cd pau-*/pau.libs/
CORRUPTED_FILES_DIR='corrupted_files/'
mkdir $CORRUPTED_FILES_DIR
CORRUPTED_FILES=`find . -maxdepth 1 -type f | grep .so | sed 's/.\///g' `
WRONG_FILENAMES=()
REQUIREMENTS=`patchelf --print-needed ../pau_cuda.cpython-36m-x86_64-linux-gnu.so`
for CORRUPTED_F in $CORRUPTED_FILES
do
  if [[ $CORRUPTED_F == *"-"* ]]; then
    WRONG_FILENAMES+=($CORRUPTED_F)
    log "--------------------------------"
    ORI_FILENAME="${CORRUPTED_F%-*}.so${CORRUPTED_F##*.so}"
    log "Found $CORRUPTED_F, searching original $ORI_FILENAME"
    if [[ `find $TORCH_LIB -name $ORI_FILENAME` ]]; then
      log "Found $ORI_FILENAME"
      cp $TORCH_LIB/$ORI_FILENAME .
      mv $CORRUPTED_F $CORRUPTED_FILES_DIR
    elif [[ `find $CUDA_LIB -name $ORI_FILENAME` ]]; then
      log "Found $ORI_FILENAME"
      cp $CUDA_LIB/$ORI_FILENAME .
      mv $CORRUPTED_F $CORRUPTED_FILES_DIR
    else
      printf "Haven't been able to locate $ORI_FILENAME\n"
      exit 1
    fi
    if [[ "${REQUIREMENTS}" =~ "$CORRUPTED_F" ]]; then
      patchelf --replace-needed $CORRUPTED_F $ORI_FILENAME ../pau_cuda.cpython-36m-x86_64-linux-gnu.so
    fi
    log "removing line \"pau.libs/$CORRUPTED_F ...\" from RECORD"
    sed -i "/pau.libs\/$CORRUPTED_F/d" ../*.dist-info/RECORD
    export SHA256=($(sha256sum $ORI_FILENAME))
    log "And adding line \"pau.libs/$ORI_FILENAME ...\" into it"
    echo "pau.libs/$ORI_FILENAME,sha256=$SHA256,`stat --printf="%s" ../pau.libs/$ORI_FILENAME`" >> ../*.dist-info/RECORD
  fi
done
cd ../../
rm pau-*-cp36-cp36m-manylinux2014_x86_64.whl
python3 -m wheel pack *  # creates the new wheel
rm -R `ls -1 -d */`  # removes the pau directory only
cd ../

unset $TORCH_LIB #
unset $CUDA_LIB

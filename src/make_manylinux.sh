docker run -ti --gpus all --name manyl_cuda101 -v `pwd`:/prauper_src soumith/manylinux-cuda101:latest bash
# docker run -ti --gpus all --name manyl_cuda101 -v `pwd`:/prauper_src soumith/manylinux-cuda100:latest bash


# install python3.7
yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar xzf Python-3.7.9.tgz
cd Python-3.7.9
./configure --enable-optimizations
make altinstall

#install the requirements for different pythons
python3.6 -m pip install -U pip
python3.6 -m pip install -r requirements.txt

yum install -y python3-devel.x86_64
# generate the wheels
python3.6 setup.py bdist_wheel
python3.7 setup.py bdist_wheel


# use auditwheel to repair
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64/python3.6/site-packages/torch/lib/  # for it to be able to find the .so files
auditwheel -v repair --plat manylinux2014_x86_64 pau-0.0.16-cp37-cp37m-linux_x86_64.whl

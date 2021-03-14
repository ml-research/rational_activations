FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

USER root
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

RUN conda create -y -n cicd_env_cuda10.1py3.7 python=3.7
RUN conda init bash
# Make RUN commands use the new environment (better than to use conda activate, see https://pythonspeed.com/articles/activate-conda-dockerfile/):
SHELL ["conda", "run", "-n", "cicd_env_cuda10.1py3.7", "/bin/bash", "-c"]
# make conda activate command available from /bin/bash --interative sqhells
RUN conda install -y -c pytorch cudatoolkit=10.1
RUN conda install -c conda-forge cartopy
RUN pwd
RUN ls -l
# Copies your code file from your action repository to the filesystem path `/` of the container
COPY .github/workflows/docker-entrypoint.sh /docker-entrypoint.sh
# RUN ["chmod", "+x", "/docker-entrypoint.sh"]
# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/docker-entrypoint.sh"]
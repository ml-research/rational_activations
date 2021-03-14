FROM nvidia/cuda:10.1-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.6
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

RUN conda create -y -n cicd_env_cuda10.1py3.7 python=3.7
RUN conda init bash
# Make RUN commands use the new environment (better than to use conda activate, see https://pythonspeed.com/articles/activate-conda-dockerfile/):
SHELL ["conda", "run", "-n", "cicd_env_cuda10.1py3.7", "/bin/bash", "-c"]
# make conda activate command available from /bin/bash --interative shells
RUN conda install -y -c pytorch cudatoolkit=10.1

ENTRYPOINT ["conda", "run", "-n", "cicd_env_cuda10.1py3.7", "python3.7"]
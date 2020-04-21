FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

#RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh  && \
#	chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

#RUN /opt/conda/bin/conda update -y conda
#RUN /opt/conda/bin/conda clean -ya

#ENV PATH /opt/conda/bin:$PATH

#RUN conda create -y -n condaenv36 python=3.6 pytorch
#RUN conda create -y -n condaenv37 python=3.7 pytorch
#RUN conda create -y -n condaenv38 python=3.8 pytorch

#RUN conda init bash

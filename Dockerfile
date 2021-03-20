FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime 
RUN 	apt-get update -y && apt-get install -y python3-pip

RUN	mkdir /.install
COPY	requirements.txt /.install
WORKDIR /.install

RUN	pip3 install --upgrade pip
RUN	pip3 install -r requirements.txt

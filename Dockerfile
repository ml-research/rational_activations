FROM nvidia/cuda:11.2.0-devel
RUN 	apt-get update -y && apt-get install -y python3-pip

RUN	mkdir /.install
COPY	requirements.txt /.install
WORKDIR /.install

RUN	pip3 install --upgrade pip
RUN	pip3 install -r requirements.txt

SRC = AIML-TUDA

DOCKER_TAG = latest
DOCKER_TEST_IMAGE_NAME = rational_manylinux:$(DOCKER_TAG)
# DOCKER_TORCH_VERSION = 'torch==1.7.1'
# DOCKER_TEST_TORCH_VERSION = 'torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html'
# DOCKER_RUN_CMD = docker run -i --gpus device=all --name rat_manylinux -v $(pwd):/rational_activations df31f4268b9b zsh

ifeq ($(shell uname),Darwin)
ifeq ($(shell which gsed),)
$(error Please install GNU sed with 'brew install gnu-sed')
else
SED = gsed
endif
else
SED = sed
endif

.PHONY : docker-image
docker-image :
	docker build \
		--pull \
		-f Dockerfile \
		--build-arg TORCH=$(DOCKER_TORCH_VERSION) \
		-t $(DOCKER_IMAGE_NAME) .

.PHONY: docker-run
docker-run:
	$(DOCKER_RUN_CMD) --gpus all $(DOCKER_IMAGE_NAME) $(ARGS)

.PHONY: docker-test-image
docker-test-image:
	docker build --no-cache\
		-f Dockerfile.test \
		-t $(DOCKER_TEST_IMAGE_NAME) .

.PHONY : docker-test-run
docker-test-run :
	docker run --gpus device=all $(DOCKER_TEST_IMAGE_NAME)
	nvcc --version
	echo $(CUDA_HOME)
	pwd
	ls -l
	python -c "import torch; print('Cuda available:', torch.cuda.is_available())"
	python -c "import torch; print('Number of GPUs available:', torch.cuda.device_count(), 'CUDA version:', torch.version.cuda)"
	python -m pytest

.PHONY: docker-test-run-zsh
docker-test-run-zsh:
	docker run -i --gpus device=all --name rat_manylinux -v $(pwd):/rational_activations df31f4268b9b zsh
	alias nvcc="/usr/local/cuda-10.2/bin/nvcc"
	export PATH="/usr/local/cuda-10.2/bin/"
	export CUDA_HOME="/usr/local/cuda-10.2/"
	nvidia-smi
	python -c "import torch; print('Cuda available:', torch.cuda.is_available())"
	python -c "import torch; print('Number of GPUs available:', torch.cuda.device_count(), 'CUDA version:', torch.version.cuda)"
	nvcc --version
	echo $CUDA_HOME
	python setup.py develop --user
	python -m pytest


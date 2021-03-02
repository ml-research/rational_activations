SRC = AIML-TUDA

DOCKER_TAG = latest
DOCKER_TEST_IMAGE_NAME = rational_manylinux:$(DOCKER_TAG)
REPO_ROOT = ../../


.PHONY: docker-test-run-zsh
docker-test-run-zsh:
	docker run -i --gpus device=all --rm --name rat_manylinux_$(CUDA)_$(Python) rational_manylinux
	python -c "import torch; print('Cuda available:', torch.cuda.is_available())"
	python -c "import sys; print('Python version:', sys.version)"
	python -c "import torch; print('Number of GPUs available:', torch.cuda.device_count(), 'CUDA version:', torch.version.cuda)"
	cd /usr/local/ && ls -l
	cd /usr/local/cuda/ && ls -l
	nvidia-smi
	PATH=/usr/local/cuda-$(CUDA)/bin:$$PATH && nvcc --version && \
     CUDA_HOME=/usr/local/cuda-$(CUDA)/ && echo CUDA_HOME: $$CUDA_HOME && \
     cd $(REPO_ROOT) && \
	 python setup.py develop --user && \
	 python -m pytest



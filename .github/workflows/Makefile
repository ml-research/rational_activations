SRC = AIML-TUDA

DOCKER_TAG = latest
DOCKER_TEST_IMAGE_NAME = rational_manylinux:$(DOCKER_TAG)
REPO_ROOT = ../../
# DOCKER_TEST_TORCH_VERSION = 'torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html'
# DOCKER_RUN_CMD = docker run -i --gpus device=all --name rat_manylinux -v $(pwd):/rational_activations df31f4268b9b zsh


.PHONY: docker-test-run-zsh
docker-test-run-zsh:
	docker run -i --gpus device=all --rm --name rationals_cicd_cuda$(CUDA)py$(Python) cuda$(CUDA)py$(Python) bash && \
	 echo CUDA_HOME: $$CUDA_HOME && \
	 PATH=/usr/local/cuda-$(CUDA)/bin:$$PATH && nvcc --version && \
	 cd $(REPO_ROOT) && \
	 python -c "import networkx; import snorkel; print('PSSSSSS:')" && \
	 python -c "import sys; print('Python version:', sys.version)" && \
	 python -c "import torch; print('CUDA version:', torch.version.cuda)" && \
	 python$(Python) -c "import sys; print('Python version:', sys.version)" && \
	 python$(Python) -c "import torch; print('CUDA version:', torch.version.cuda)" && \
	 pip$(Python) install -r requirements.txt && \
	 python$(Python) setup.py develop --user && \
	 python$(Python) -m pytest



.PHONY: docker-test-run-zshOLD
docker-test-run-zshOLD:
	docker run -i --gpus device=all --rm --name rat_manylinux_$(CUDA)_$(Python) rational_manylinux
	python -c "import torch; print('Cuda available:', torch.cuda.is_available())"
	python -c "import sys; print('Python version:', sys.version)"
	python -c "import torch; print('Number of GPUs available:', torch.cuda.device_count(), 'CUDA version:', torch.version.cuda)"
	nvcc --version
	echo CUDA_HOME: $$CUDA_HOME
	echo PATH: $$PATH
	cd /usr/local/ && ls -l
	cd /usr/local/cuda/ && ls -l
	nvidia-smi
	PATH=/usr/local/cuda-$(CUDA)/bin:$$PATH && nvcc --version && \
	 cd $(REPO_ROOT) && \
	 python setup.py develop --user && \
	 python -m pytest



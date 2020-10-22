import torch

from pau_torch.pade_activation_unit import PAU
import numpy as np

t = [-2., -1, 0., 1., 2.]
expected_res = np.array([-0.02, -0.01, 0, 1, 2])
inp = torch.from_numpy(np.array(t)).reshape(-1)
cuda_inp = torch.tensor(np.array(t), dtype=torch.float, device="cuda").reshape(-1)


pauA_lrelu_cpu = PAU(version='A', cuda=False)(inp).detach().numpy()
pauB_lrelu_cpu = PAU(version='B', cuda=False)(inp).detach().numpy()
pauC_lrelu_cpu = PAU(version='C', cuda=False)(inp).detach().numpy()
pauD_lrelu_cpu = PAU(version='D', cuda=False, trainable=False)(inp).detach().numpy()

pauA_lrelu_gpu = PAU(version='A', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
pauB_lrelu_gpu = PAU(version='B', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
pauC_lrelu_gpu = PAU(version='C', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
pauD_lrelu_gpu = PAU(version='D', cuda=True, trainable=False)(cuda_inp).clone().detach().cpu().numpy()


#  Tests on cpu
def test_pauA_cpu_lrelu():
    assert np.all(np.isclose(pauA_lrelu_cpu, expected_res, atol=5e-02))


def test_pauB_cpu_lrelu():
    assert np.all(np.isclose(pauB_lrelu_cpu, expected_res, atol=5e-02))


def test_pauC_cpu_lrelu():
    assert np.all(np.isclose(pauC_lrelu_cpu, expected_res, atol=5e-02))


def test_pauD_cpu_lrelu():
    assert np.all(np.isclose(pauD_lrelu_cpu, expected_res, atol=5e-02))
    # print(pauD_lrelu_cpu)


# Tests on GPU
def test_pauA_gpu_lrelu():
    assert np.all(np.isclose(pauA_lrelu_gpu, expected_res, atol=5e-02))


def test_pauB_gpu_lrelu():
    assert np.all(np.isclose(pauB_lrelu_gpu, expected_res, atol=5e-02))


def test_pauC_gpu_lrelu():
    assert np.all(np.isclose(pauC_lrelu_gpu, expected_res, atol=5e-02))


def test_pauD_gpu_lrelu():
    assert np.all(np.isclose(pauD_lrelu_gpu, expected_res, atol=5e-02))


# GPU and CPU consistent results
def test_cpu_equal_gpu_A():
    assert np.all(np.isclose(pauA_lrelu_cpu, pauA_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_B():
    assert np.all(np.isclose(pauB_lrelu_cpu, pauB_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_C():
    assert np.all(np.isclose(pauC_lrelu_cpu, pauC_lrelu_gpu, atol=1e-06))

def test_cpu_equal_gpu_D():
    assert np.all(np.isclose(pauD_lrelu_cpu, pauD_lrelu_gpu, atol=1e-06))


# Tests conversion GPU -> CPU
def test_conversion_gpu_to_cpuA():
    pau = PAU(version='A', cuda=True)
    pau.cpu()
    params = np.all([str(para.device) == 'cpu' for para in pau.parameters()])
    cpu_f = "PYTORCH_A" in pau.activation_function.__qualname__
    new_res = pau(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_gpu_to_cpuB():
    pau = PAU(version='B', cuda=True)
    pau.cpu()
    params = np.all([str(para.device) == 'cpu' for para in pau.parameters()])
    cpu_f = "PYTORCH_B" in pau.activation_function.__qualname__
    new_res = pau(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_gpu_to_cpuC():
    pau = PAU(version='C', cuda=True)
    pau.cpu()
    params = np.all([str(para.device) == 'cpu' for para in pau.parameters()])
    cpu_f = "PYTORCH_C" in pau.activation_function.__qualname__
    new_res = pau(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_gpu_to_cpuD():
    pau = PAU(version='D', cuda=True, trainable=False)
    pau.cpu()
    params = np.all([str(para.device) == 'cpu' for para in pau.parameters()])
    cpu_f = "PYTORCH_D" in pau.activation_function.__qualname__
    new_res = pau(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


# Tests conversion CPU -> GPU
def test_conversion_cpu_to_gpuA():
    pau = PAU(version='A', cuda=False)
    pau.cuda()
    params = np.all(['cuda' in str(para.device) for para in pau.parameters()])
    cpu_f = "CUDA_A" in pau.activation_function.__qualname__
    new_res = pau(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_cpu_to_gpuB():
    pau = PAU(version='B', cuda=False)
    pau.cuda()
    params = np.all(['cuda' in str(para.device) for para in pau.parameters()])
    cpu_f = "CUDA_B" in pau.activation_function.__qualname__
    new_res = pau(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_cpu_to_gpuC():
    pau = PAU(version='C', cuda=False)
    pau.cuda()
    params = np.all(['cuda' in str(para.device) for para in pau.parameters()])
    cpu_f = "CUDA_C" in pau.activation_function.__qualname__
    new_res = pau(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_cpu_to_gpuD():
    pau = PAU(version='D', cuda=False, trainable=False)
    pau.cuda()
    params = np.all(['cuda' in str(para.device) for para in pau.parameters()])
    cpu_f = "CUDA_D" in pau.activation_function.__qualname__
    new_res = pau(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute

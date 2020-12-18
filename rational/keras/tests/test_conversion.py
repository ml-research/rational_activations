import tensorflow as tf

from rational.keras import Rational
import numpy as np


# MODIFY EVERY THING SUCH THAT IT TEST ON RATIONALS OF KERAS


t = [-2., -1, 0., 1., 2.]
expected_res = np.array([-0.02, -0.01, 0, 1, 2])
inp = tf.convert_to_tensor(np.array(t, np.float32), np.float32)
cuda_inp = tf.convert_to_tensor(expected_res, np.float32)


rationalA_lrelu_cpu = Rational(version='A', cuda=False)(inp).numpy()
rationalB_lrelu_cpu = Rational(version='B', cuda=False)(inp).numpy()
rationalC_lrelu_cpu = Rational(version='C', cuda=False)(inp).numpy()
#rationalD_lrelu_cpu = Rational(version='D', cuda=False, trainable=False)(inp).numpy()

#rationalA_lrelu_gpu = Rational(version='A', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
#rationalB_lrelu_gpu = Rational(version='B', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
#rationalC_lrelu_gpu = Rational(version='C', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
#rationalD_lrelu_gpu = Rational(version='D', cuda=True, trainable=False)(cuda_inp).clone().detach().cpu().numpy()

# GPU and CPU consistent results

'''
def test_cpu_equal_gpu_A():
    assert np.all(np.isclose(rationalA_lrelu_cpu,
                             rationalA_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_B():
    assert np.all(np.isclose(rationalB_lrelu_cpu,
                             rationalB_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_C():
    assert np.all(np.isclose(rationalC_lrelu_cpu,
                             rationalC_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_D():
    assert np.all(np.isclose(rationalD_lrelu_cpu,
                             rationalD_lrelu_gpu, atol=1e-06))


# Tests conversion GPU -> CPU
def test_conversion_gpu_to_cpuA():
    rational = Rational(version='A', cuda=True)
    rational.cpu()
    params = np.all(
        [str(para.device) == 'cpu' for para in rational.parameters()])
    cpu_f = "PYTORCH_A" in rational.activation_function.__qualname__
    new_res = rational(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_gpu_to_cpuB():
    rational = Rational(version='B', cuda=True)
    rational.cpu()
    params = np.all(
        [str(para.device) == 'cpu' for para in rational.parameters()])
    cpu_f = "PYTORCH_B" in rational.activation_function.__qualname__
    new_res = rational(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_gpu_to_cpuC():
    rational = Rational(version='C', cuda=True)
    rational.cpu()
    params = np.all(
        [str(para.device) == 'cpu' for para in rational.parameters()])
    cpu_f = "PYTORCH_C" in rational.activation_function.__qualname__
    new_res = rational(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_gpu_to_cpuD():
    rational = Rational(version='D', cuda=True, trainable=False)
    rational.cpu()
    params = np.all(
        [str(para.device) == 'cpu' for para in rational.parameters()])
    cpu_f = "PYTORCH_D" in rational.activation_function.__qualname__
    new_res = rational(inp).detach().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


# Tests conversion CPU -> GPU
def test_conversion_cpu_to_gpuA():
    rational = Rational(version='A', cuda=False)
    rational.cuda()
    params = np.all(['cuda' in str(para.device)
                     for para in rational.parameters()])
    cpu_f = "CUDA_A" in rational.activation_function.__qualname__
    new_res = rational(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_cpu_to_gpuB():
    rational = Rational(version='B', cuda=False)
    rational.cuda()
    params = np.all(['cuda' in str(para.device)
                     for para in rational.parameters()])
    cpu_f = "CUDA_B" in rational.activation_function.__qualname__
    new_res = rational(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_cpu_to_gpuC():
    rational = Rational(version='C', cuda=False)
    rational.cuda()
    params = np.all(['cuda' in str(para.device)
                     for para in rational.parameters()])
    cpu_f = "CUDA_C" in rational.activation_function.__qualname__
    new_res = rational(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute


def test_conversion_cpu_to_gpuD():
    rational = Rational(version='D', cuda=False, trainable=False)
    rational.cuda()
    params = np.all(['cuda' in str(para.device)
                     for para in rational.parameters()])
    cpu_f = "CUDA_D" in rational.activation_function.__qualname__
    new_res = rational(cuda_inp).clone().detach().cpu().numpy()
    coherent_compute = np.all(np.isclose(new_res, expected_res, atol=5e-02))
    assert params and cpu_f and coherent_compute
'''

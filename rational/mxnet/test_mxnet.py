from mxnet import nd, gpu, cpu

from rational.mxnet import Rational
import numpy as np

t = [-2., -1, 0., 1., 2.]
expected_res = np.array([-0.02, -0.01, 0, 1, 2])
inp = nd.array(np.array(t)).reshape(-1)
cuda_inp = nd.array(np.array(t), ctx=gpu()).reshape(-1)

rationalA_lrelu_cpu = Rational(version='A', device=cpu())(inp).asnumpy()
rationalB_lrelu_cpu = Rational(version='B', device=cpu())(inp).asnumpy()
rationalC_lrelu_cpu = Rational(version='C', device=cpu())(inp).asnumpy()
rationalD_lrelu_cpu = Rational(version='D', device=cpu(), trainable=False)(inp).asnumpy()

rationalA_lrelu_gpu = Rational(version='A', device=gpu())(cuda_inp).clone().copyto(cpu()).asnumpy()
rationalB_lrelu_gpu = Rational(version='B', device=gpu())(cuda_inp).clone().copyto(cpu()).asnumpy()
rationalC_lrelu_gpu = Rational(version='C', device=gpu())(cuda_inp).clone().copyto(cpu()).asnumpy()
rationalD_lrelu_gpu = Rational(version='D', device=gpu(), trainable=False)(cuda_inp).clone().copyto(cpu()).asnumpy()


#  Tests on cpu
def test_rationalA_cpu_lrelu():
    assert np.all(np.isclose(rationalA_lrelu_cpu, expected_res, atol=5e-02))


def test_rationalB_cpu_lrelu():
    assert np.all(np.isclose(rationalB_lrelu_cpu, expected_res, atol=5e-02))


def test_rationalC_cpu_lrelu():
    assert np.all(np.isclose(rationalC_lrelu_cpu, expected_res, atol=5e-02))


def test_rationalD_cpu_lrelu():
    assert np.all(np.isclose(rationalD_lrelu_cpu, expected_res, atol=5e-02))
    # print(rationalD_lrelu_cpu)


# Tests on GPU
def test_rationalA_gpu_lrelu():
    assert np.all(np.isclose(rationalA_lrelu_gpu, expected_res, atol=5e-02))


def test_rationalB_gpu_lrelu():
    assert np.all(np.isclose(rationalB_lrelu_gpu, expected_res, atol=5e-02))


def test_rationalC_gpu_lrelu():
    assert np.all(np.isclose(rationalC_lrelu_gpu, expected_res, atol=5e-02))


def test_rationalD_gpu_lrelu():
    assert np.all(np.isclose(rationalD_lrelu_gpu, expected_res, atol=5e-02))


# GPU and CPU consistent results
def test_cpu_equal_gpu_A():
    assert np.all(np.isclose(rationalA_lrelu_cpu, rationalA_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_B():
    assert np.all(np.isclose(rationalB_lrelu_cpu, rationalB_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_C():
    assert np.all(np.isclose(rationalC_lrelu_cpu, rationalC_lrelu_gpu, atol=1e-06))


def test_cpu_equal_gpu_D():
    assert np.all(np.isclose(rationalD_lrelu_cpu, rationalD_lrelu_gpu, atol=1e-06))

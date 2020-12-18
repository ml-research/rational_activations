import numpy as np
import tensorflow as tf

from rational.keras import Rational
from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid

# initialization of arrays
t = [-2., -1, 0., 1., 2.]
expected_res_lrelu = np.array(leaky_relu(t, alpha=0.01))
expected_res_tanh = np.array(tanh(t))
expected_res_sigmoid = np.array(sigmoid(t))
inp = tf.convert_to_tensor(np.array(t, np.float32), np.float32)

# initialization of Rational LReLU in CPU
rationalA_lrelu_cpu = Rational(version='A', cuda=False)(inp).numpy()
rationalB_lrelu_cpu = Rational(version='B', cuda=False)(inp).numpy()
rationalC_lrelu_cpu = Rational(version='C', cuda=False)(inp).numpy()
#rationalD_lrelu_cpu = Rational(version='D', cuda=False)(inp).numpy()

# initialization of Rational tanh in CPU
rationalA_tanh_cpu = Rational('tanh', version='A', cuda=False)(inp).numpy()
rationalB_tanh_cpu = Rational('tanh', version='B', cuda=False)(inp).numpy()
rationalC_tanh_cpu = Rational('tanh', version='C', cuda=False)(inp).numpy()
#rationalD_tanh_cpu = Rational('tanh', version='D', cuda=False)(inp).numpy()

# initialization of Rational sigmoid in CPU
rationalA_sigmoid_cpu = Rational(
    'sigmoid', version='A', cuda=False)(inp).numpy()
rationalB_sigmoid_cpu = Rational(
    'sigmoid', version='B', cuda=False)(inp).numpy()
rationalC_sigmoid_cpu = Rational(
    'sigmoid', version='C', cuda=False)(inp).numpy()
#rationalC_sigmoid_cpu = Rational('sigmoid', version='D', cuda=False)(inp).numpy()


# tests on approximation of lrelu
def test_rationalA_cpu_lrelu():
    assert np.all(np.isclose(rationalA_lrelu_cpu,
                             expected_res_lrelu, atol=5e-02))


def test_rationalB_cpu_lrelu():
    assert np.all(np.isclose(rationalB_lrelu_cpu,
                             expected_res_lrelu, atol=5e-02))


def test_rationalC_cpu_lrelu():
    assert np.all(np.isclose(rationalC_lrelu_cpu,
                             expected_res_lrelu, atol=5e-02))


# def test_rationalD_cpu_lrelu():
    #assert np.all(np.isclose(rationalD_lrelu_cpu, expected_res_lrelu, atol=5e-02))


# tests on approximation of tanh
def test_rationalA_cpu_tanh():
    assert np.all(np.isclose(rationalA_tanh_cpu,
                             expected_res_tanh, atol=5e-02))


def test_rationalB_cpu_tanh():
    assert np.all(np.isclose(rationalB_tanh_cpu,
                             expected_res_tanh, atol=5e-02))


def test_rationalC_cpu_tanh():
    assert np.all(np.isclose(rationalC_tanh_cpu,
                             expected_res_tanh, atol=5e-02))


# def test_rationalD_cpu_tanh():
    #assert np.all(np.isclose(rationalD_tanh_cpu, expected_res_tanh, atol=5e-02))


# tests on approximation of sigmoid
def test_rationalA_cpu_sigmoid():
    assert np.all(np.isclose(rationalA_sigmoid_cpu,
                             expected_res_sigmoid, atol=5e-02))


def test_rationalB_cpu_sigmoid():
    assert np.all(np.isclose(rationalB_sigmoid_cpu,
                             expected_res_sigmoid, atol=5e-02))


def test_rationalC_cpu_sigmoid():
    assert np.all(np.isclose(rationalC_sigmoid_cpu,
                             expected_res_sigmoid, atol=5e-02))


# def test_rationalD_cpu_sigmoid():
    #assert np.all(np.isclose(rationalD_tsigmoid_cpu, expected_res_sigmoid, atol=5e-02))

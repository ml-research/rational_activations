import numpy as np
import tensorflow as tf

from rational.keras import Rational
from tensorflow.nn import leaky_relu
from tensorflow.math import tanh, sigmoid

# MODIFY EVERY THING SUCH THAT IT TEST ON RATIONALS OF KERAS

t = [-2., -1, 0., 1., 2.]
expected_res_lrelu = np.array(leaky_relu(t, alpha=0.01))
expected_res_tanh = np.array(tanh(t))
expected_res_sigmoid = np.array(sigmoid(t))
inp = tf.convert_to_tensor(np.array(t, np.float32), np.float32)
#cuda_inp = tf.convert_to_tensor(expected_res, np.float32)

rationalA_lrelu_cpu = Rational(version='A', cuda=False)(inp).numpy()
rationalB_lrelu_cpu = Rational(version='B', cuda=False)(inp).numpy()
rationalC_lrelu_cpu = Rational(version='C', cuda=False)(inp).numpy()
#rationalD_lrelu_cpu = Rational(version='D', cuda=False)(inp).numpy()

rationalA_tanh_cpu = Rational('tanh', version='A', cuda=False)(inp).numpy()
rationalB_tanh_cpu = Rational('tanh', version='B', cuda=False)(inp).numpy()
rationalC_tanh_cpu = Rational('tanh', version='C', cuda=False)(inp).numpy()
#rationalD_tanh_cpu = Rational('tanh', version='D', cuda=False)(inp).numpy()

rationalA_sigmoid_cpu = Rational(
    'sigmoid', version='A', cuda=False)(inp).numpy()
rationalB_sigmoid_cpu = Rational(
    'sigmoid', version='B', cuda=False)(inp).numpy()
rationalC_sigmoid_cpu = Rational(
    'sigmoid', version='C', cuda=False)(inp).numpy()
#rationalC_sigmoid_cpu = Rational('sigmoid', version='D', cuda=False)(inp).numpy()

# todo: lrelu, tanh, sigmoid, ...)


# tests on approximation of lrelu
def test_rationalA_cpu_lrelu():
    print("rational_lrelu result: ", rationalA_lrelu_cpu)
    print("lrelu result: ", expected_res)
    assert (rationalA_lrelu_cpu == expected_res_lrelu).all()


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
    print("rational_tanh result: ", rationalA_tanh_cpu)
    print("lrelu result: ", expected_res_tanh)
    assert (rationalA_tanh_cpu == expected_res_tanh).all()


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
    print("rational_sigmoid result: ", rationalA_sigmoid_cpu)
    print("sigmoid result: ", expected_res_sigmoid)
    assert (rationalA_sigmoid_cpu == expected_res_sigmoid).all()


def test_rationalB_cpu_sigmoid():
    assert np.all(np.isclose(rationalB_sigmoid_cpu,
                             expected_res_sigmoid, atol=5e-02))


def test_rationalC_cpu_sigmoid():
    assert np.all(np.isclose(rationalC_sigmoid_cpu,
                             expected_res_sigmoid, atol=5e-02))


# def test_rationalD_cpu_sigmoid():
    #assert np.all(np.isclose(rationalD_tsigmoid_cpu, expected_res_sigmoid, atol=5e-02))
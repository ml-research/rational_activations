import tensorflow as tf

from rational.keras import Rational
import numpy as np


### MODIFY EVERY THING SUCH THAT IT TEST ON RATIONALS OF KERAS


t = [-2., -1, 0., 1., 2.]
expected_res = np.array([-0.02, -0.01, 0, 1, 2])
inp = tf.convert_to_tensor(np.array(t, np.float32), np.float32)
cuda_inp = tf.convert_to_tensor(expected_res, np.float32)


rationalA_lrelu_cpu = Rational(version='A', cuda=False)(inp).numpy()
rationalB_lrelu_cpu = Rational(version='B', cuda=False)(inp).numpy()
rationalC_lrelu_cpu = Rational(version='C', cuda=False)(inp).numpy()
#rationalD_lrelu_cpu = Rational(version='D', cuda=False, trainable=False)(inp).numpy()

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


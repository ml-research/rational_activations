import tensorflow as tf

import numpy as np


### MODIFY EVERY THING SUCH THAT IT TEST ON RATIONALS OF KERAS


t = [-2., -1, 0., 1., 2.]
expected_res = np.array([-0.02, -0.01, 0, 1, 2])
inp = tf.convert_to_tensor(np.array(t, np.float32), np.float32)
cuda_inp = tf.convert_to_tensor(expected_res, np.float32)

#rationalA_lrelu_gpu = Rational(version='A', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
#rationalB_lrelu_gpu = Rational(version='B', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
#rationalC_lrelu_gpu = Rational(version='C', cuda=True)(cuda_inp).clone().detach().cpu().numpy()
#rationalD_lrelu_gpu = Rational(version='D', cuda=True, trainable=False)(cuda_inp).clone().detach().cpu().numpy()

# Tests on GPU
def test_rationalA_gpu_lrelu():
    assert np.all(np.isclose(rationalA_lrelu_gpu, expected_res, atol=5e-02))


def test_rationalB_gpu_lrelu():
    assert np.all(np.isclose(rationalB_lrelu_gpu, expected_res, atol=5e-02))


def test_rationalC_gpu_lrelu():
    assert np.all(np.isclose(rationalC_lrelu_gpu, expected_res, atol=5e-02))


def test_rationalD_gpu_lrelu():
    assert np.all(np.isclose(rationalD_lrelu_gpu, expected_res, atol=5e-02))
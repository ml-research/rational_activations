import tensorflow as tf

from rational_keras import Rational
import numpy as np


### MODIFY EVERY THING SUCH THAT IT TEST ON RATIONALS OF KERAS


t = [-2., -1, 0., 1., 2.]
expected_res = np.array([-0.02, -0.01, 0, 1, 2])
inp = tf.convert_to_tensor(np.array(t, np.float32), np.float32)
cuda_inp = tf.convert_to_tensor(expected_res, np.float32)

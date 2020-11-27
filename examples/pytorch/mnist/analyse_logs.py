import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='dataset', default='cifar10'),
parser.add_argument('--aggregation', type=str, help='aggregation', default='mean'),
parser.add_argument('--arch', type=str, help='networks to use', default='vgg8,mobilenetv2,densenet121,resnet101'),

args = parser.parse_args()


def get_data(log_file, tag_name):
    result = []
    for e in tf.compat.v1.train.summary_iterator(log_file):
        for v in e.summary.value:
            if v.tag == tag_name:
                result.append(v.simple_value)

    return np.array(result)


log_tags = [
    "test/accuracy",
    "test/loss",
    "train/loss",
    "train/loss_epoch"
]
experiments = ['pau', 'recurrent_pau', 'relu']

res = {}
for e in experiments:
    res[e] = {}
    for tag in log_tags:
        log_file_path_pade = ".examples/experiments/missing_mnist/paper_mnist_vgg_sgd_seed31/{}/".format(e)
        log_file_pade = glob.glob(log_file_path_pade + '/events.out.tfevents.*')
        data = get_data(log_file_pade[0], tag)
        res[e][tag] = data
pickle.dump(res, open(".examples/runs/missing_mnist/paper_mnist_vgg_sgd_seed31/mnist_vgg_seed31.quention", "wb"))
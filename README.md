# Rational Activations - Learnable Rational Activation Functions
First introduce as PAU in Padé Activation Units: End-to-end Learning of Activation Functions in Deep Neural Network

Arxiv link: https://arxiv.org/abs/1907.06732

## 1. About Padé Activation Units

Rational Activations are a novel learnable activation functions. Rationals encode activation functions as rational functions, trainable in an end-to-end fashion using backpropagation and can be seemingless integrated into any neural network in the same way as common activation functions (e.g. ReLU).

<table border="0">
<tr>
    <td>
    <img src="./images/results.png" width="100%" />
    </td>
</tr>
</table>

Rational matches or outperforms common activations in terms of predictive performance and training time.
And, therefore relieves the network designer of having to commit to a potentially underperforming choice.

## 2. Dependencies
    PyTorch>=1.4.0
    CUDA>=10.1


## 3. Installation

To install the rational_activations module, you can use pip, but you should be careful about the CUDA version running on your machine.
To get your CUDA version:
    import torch
    torch.version.cuda


<center><iframe src="tableau.html" width="800" height="320" seamless ></iframe ></center>


    pip3 install wheel

For CUDA 10.1 (and thus 1.4.0>=torch>= 1.5.0), download the wheel corresponding to your python3 version in the _wheelhouse_ repo and install it with:

    pip3 install rational-0.0.16-101-cp{your_version}-manylinux2014_x86_64.whl

If you encounter any trouble installing rational, please contact [this person](quentin.delfosse@cs.tu-darmstadt.de).

## 4. Using Rational in Neural Networks

Rational can be integrated in the same way as any other common activation function.

~~~~
import torch
from rational_torch import Rational

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    Rational(), # e.g. instead of torch.nn.ReLU()
    torch.nn.Linear(H, D_out),
)
~~~~

## 5. Reproducing Results

To reproduce the reported results of the paper execute:

$ export PYTHONPATH="./"
$ python experiments/main.py --dataset mnist --arch conv --optimizer adam --lr 2e-3

    # DATASET: Name of the dataset, for MNIST use mnist and for Fashion-MNIST use fmnist
    # ARCH: selected neural network architecture: vgg, lenet or conv
    # OPTIMIZER: either adam or sgd
    # LR: learning rate


## 6. To be implemented
- [X] Make a documentation
- [X] Create tutorial in the doc
- [ ] Tensorflow working version
- [ ] Automatically find initial approx weights for function list
- [ ] Repair + enhance Automatic manylinux production script.
- [ ] Add python3.9 support
- [ ] Make an CUDA 11.0 compatible version
- [ ] Repair the tox test and have them checking before commit

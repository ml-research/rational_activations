<span style="white-space: nowrap;">![Logo](./images/rationals_logo_colored.png)</span><h1>Rational Activations - Learnable Rational Activation Functions</h1>
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

To install the rational_activations module, you can use pip, but:<br/>

:bangbang:  You should be careful about the CUDA version running on your machine.


To get your CUDA version:

    import torch
    torch.version.cuda

For **your** corresponding version of CUDA, please use one of the following command blocks:
### CUDA 10.2 (Pytorch >= 1.5.0)

     pip3 install -U pip wheel
     pip3 install torch rational-activations

### CUDA 10.1 (Pytorch == 1.4.0)
#### Python3.6

       pip3 install -U pip wheel
       pip3 install torch==1.4.0
       pip3 install https://iron.aiml.informatik.tu-darmstadt.de/wheelhouse/cuda-10.1/rational_activations-0.0.19-cp36-cp36m-manylinux2014_x86_64.whl

#### Python3.7

       pip3 install -U pip wheel
       pip3 install torch==1.4.0
       pip3 install https://iron.aiml.informatik.tu-darmstadt.de/wheelhouse/cuda-10.1/rational_activations-0.0.19-cp37-cp37m-manylinux2014_x86_64.whl

#### Python3.8

         pip3 install -U pip wheel
         pip3 install torch==1.4.0
         pip3 install https://iron.aiml.informatik.tu-darmstadt.de/wheelhouse/cuda-10.1/rational_activations-0.0.19-cp38-cp38-manylinux2014_x86_64.whl


### Other CUDA/Pytorch</h3>
For any other combinaison of python, please install from source:

     pip3 install airspeed
     git clone https://github.com/ml-research/rational_activations.git
     cd rational_activations
     python3 setup.py install --user



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


## 5. To be implemented
- [X] Make a documentation
- [X] Create tutorial in the doc
- [ ] Tensorflow working version
- [ ] Automatically find initial approx weights for function list
- [ ] Repair + enhance Automatic manylinux production script.
- [ ] Add python3.9 support
- [ ] Make an CUDA 11.0 compatible version
- [ ] Repair the tox test and have them checking before commit

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

    <style>
    .tab {
     overflow: hidden;
     border: 1px solid #ccc;
     background-color: #f1f1f1;
    }

    .tab button {
     background-color: inherit;
     float: left;
     border: none;
     outline: none;
     cursor: pointer;
     padding: 14px 16px;
     transition: 0.3s;
    }

    .tab button:hover {
     background-color: #ddd;
    }

    .tab button.active {
     background-color: #ccc;
    }

    .tabcontent {
     display: none;
     padding: 6px 12px;
     border: 1px solid #ccc;
     border-top: none;
    }

    </style>
    <script>
    function openTab(evt, tabName) {
      // Declare all variables
      var i, tabcontent, tablinks;

      // Get all elements with class="tabcontent" and hide them
      tabcontent = document.getElementsByClassName("tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }

      // Get all elements with class="tablinks" and remove the class "active"
      tablinks = document.getElementsByClassName("tablinks");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }

      // Show the current tab, and add an "active" class to the button that opened the tab
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }


    function openSubTab(evt, subTabName) {
      // Declare all variables
      var i, subtabcontent, subtablinks;

      // Get all elements with class="subtabcontent" and hide them
      subtabcontent = document.getElementsByClassName("subtabcontent");
      for (i = 0; i < subtabcontent.length; i++) {
        subtabcontent[i].style.display = "none";
      }

      // Get all elements with class="subtablinks" and remove the class "active"
      subtablinks = document.getElementsByClassName("subtablinks");
      for (i = 0; i < subtablinks.length; i++) {
        subtablinks[i].className = subtablinks[i].className.replace(" active", "");
      }

      // Show the current tab, and add an "active" class to the button that opened the tab
      document.getElementById(subTabName).style.display = "block";
      evt.currentTarget.className += " active";
    }
    </script>


    <div class="tab">
     <button class="tablinks" onclick="openTab(event, '10.2')">10.2</button>
     <button class="tablinks" onclick="openTab(event, '10.1')">10.1</button>
     <button class="tablinks" onclick="openTab(event, 'other')">other</button>
    </div>

    <div id="10.2" class="tabcontent">
     <h3>CUDA 10.2 (Pytorch >= 1.5.0)</h3>
     <p>Please use the following commands.</p>
     <code>
       pip3 install -U pip wheel <br/>
       pip3 install torch rational-activations
     </code>
    </div>

    <div id="10.1" class="tabcontent">
      <div class="tab">
       <button class="subtablinks" onclick="openSubTab(event, 'Python3.6')">Python3.6</button>
       <button class="subtablinks" onclick="openSubTab(event, 'Python3.7')">Python3.7</button>
       <button class="subtablinks" onclick="openSubTab(event, 'Python3.8')">Python3.8</button>
      </div>
      <div id="Python3.6" class="subtabcontent">
         <h3>CUDA 10.1 (Pytorch == 1.4.0)</h3>
         <h4>Python3.6</h4>
         <p>Please use the following commands:</p>
         <code>
           pip3 install -U pip wheel <br/>
           pip3 install torch==1.4.0 <br/>
           pip3 install https://iron.aiml.informatik.tu-darmstadt.de/wheelhouse/cuda-10.1/rational_activations-0.0.18-cp36-cp36m-manylinux2014_x86_64.whl
         </code>
      </div>
      <div id="Python3.7" class="subtabcontent" style="display:none">
         <h3>CUDA 10.1 (Pytorch == 1.4.0)</h3>
         <h4>Python3.7</h4>
         <p>Please use the following commands:</p>
         <code>
           pip3 install -U pip wheel <br/>
           pip3 install torch==1.4.0 <br/>
           pip3 install https://iron.aiml.informatik.tu-darmstadt.de/wheelhouse/cuda-10.1/rational_activations-0.0.18-cp37-cp37m-manylinux2014_x86_64.whl
         </code>
      </div>
      <div id="Python3.8" class="subtabcontent" style="display:none">
         <h3>CUDA 10.1 (Pytorch == 1.4.0)</h3>
         <h4>Python3.8</h4>
         <p>Please use the following commands:</p>
         <code>
           pip3 install -U pip wheel <br/>
           pip3 install torch==1.4.0 <br/>
           pip3 install https://iron.aiml.informatik.tu-darmstadt.de/wheelhouse/cuda-10.1/rational_activations-0.0.18-cp38-cp38-manylinux2014_x86_64.whl
         </code>
      </div>
    </div>

    <div id="other" class="tabcontent">
     <h3>Other CUDA/Pytorch</h3>
     <p>For any other combinaison of python, please install from source</p>
     <code>
       pip3 install airspeed <br/>
       git clone https://github.com/ml-research/rational_activations.git <br/>
       cd rational_activations<br/>
       python3 setup.py install --user
     </code>
    </div>


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

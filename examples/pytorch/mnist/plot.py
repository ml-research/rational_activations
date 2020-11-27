import torch
import numpy as np

import pickle
torch.manual_seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(17)
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import os
from rational.torch import Rational, RecurrentRational, RecurrentRationalModule
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from examples.pytorch.mnist.mnist import VGG, LeNet5, actfvs
from matplotlib import pyplot as plt
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

torch.set_anomaly_enabled(True)
cnt = 0


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, test_loss,
                                                                                            correct,
                                                                                            len(test_loader.dataset),
                                                                                            acc))
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use')
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--init', type=str, default="", choices=["", "xavier", "he"])
    args = parser.parse_args()

    networks = dict({
        "vgg": VGG,
        "lenet": LeNet5,
    })

    network = networks[args.arch]
    activation_function_keys = [x for x in list(actfvs.keys()) if 'pau' in x]

    optimizer = 'sgd'
    epochs = [1,4,5,6,7,10,15,20,'final']
    #epochs = ['final']
    for activation_function_key in activation_function_keys:
        for epoch in epochs:
            print("---" * 42)
            print("Starting with dataset: {}, activation function: {}".format(args.dataset, activation_function_key))
            print("---" * 42)
            load_path = 'examples/runs/mnist/paper_{}_{}_{}{}_seed{}/'.format(args.dataset,
                                                                                                   args.arch,
                                                                                                   optimizer,
                                                                                                   "_init_{}".format(args.init) if args.init != "" else "",
                                                                 args.seed) + activation_function_key
            use_cuda = not args.no_cuda and torch.cuda.is_available()
            #print(torch.cuda.is_available())
            #exit()

            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(args.seed)

            device = torch.device("cuda" if use_cuda else "cpu")

            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            if args.dataset == 'mnist':
                test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
                lr_scheduler_milestones = [30, 60, 90]  # Simple CNN with 3 Conv
                # lr_scheduler_milestones = [40, 80]  # VGG
            elif args.dataset == 'fmnist':
                test_loader = torch.utils.data.DataLoader(
                    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=args.test_batch_size, shuffle=True, **kwargs)
                lr_scheduler_milestones = [40, 80]
            else:
                raise ValueError('dataset error')

            model = network(activation_func=activation_function_key).to(device)
            #model.apply(weights_init_he)
            model.load_state_dict(torch.load(os.path.join(load_path, 'model_{}.pt'.format(epoch))))

            paus = list()

            cnt = 0
            for name, layer in model.named_modules():
                if isinstance(layer, Rational):
                    layer.input_retrieve_mode(max_saves=10)
                    paus.append(('rational', name, layer))
                if isinstance(layer, RecurrentRationalModule):
                    layer.input_retrieve_mode(max_saves=10)
                    paus.append(('recurrent_rational', name, layer))
                    cnt += 1

            if len(paus) > 0:
                os.makedirs(os.path.join(load_path, 'plots'), exist_ok=True)
                print("Starting model eval")
                acc = test(args, model, device, test_loader, epoch)
                print("Finished model eval -> Plot")
                fig = plt.figure(1, figsize=(6*len(paus),6))
                for i, p in enumerate(paus):
                    plt.subplot(1, len(paus), i+1)
                    plt.title("{}_{}".format(p[0], p[1]))
                    p[2].show(display=False)
                pickle.dump(fig, open(os.path.join(load_path, 'plots',
                                                           '{}_(acc{:.2f}%).fig_quention'.format(epoch, acc)), "wb"))
                plt.savefig(os.path.join(load_path, 'plots/{}_(acc{:.2f}%).png'.format(epoch, acc)),
                            bbox_inches='tight')
                plt.close(fig)
                #fig.clf()
            else:
                print("No Rational Activations found. Exit without plotting")

if __name__ == '__main__':
    main()

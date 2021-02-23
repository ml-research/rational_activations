import torch
import numpy as np

torch.manual_seed(17)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(17)
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
import os
from rational.torch import Rational, RecurrentRational
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

writer = None
matplotlib.rc('font', **font)

torch.set_anomaly_enabled(True)
cnt = 0


actfvs = dict()


actfvs["rn"] = Rational
actfvs["relu"] = nn.ReLU

#_rational = Rational()
#def shared_Rational():
#    return RecurrentRational(_rational)


actfvs["rrn"] = RecurrentRational()


def vgg_block(num_convs, in_channels, num_channels, actv_function):
    layers = []
    for i in range(num_convs):
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1)]
        in_channels = num_channels
    layers += [actv_function()]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(0.01)
        #nn.init.xavier_normal_(m.bias.data, gain=nn.init.calculate_gain('relu'))

def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(0.01)
        #nn.init.xavier_normal_(m.bias.data, gain=nn.init.calculate_gain('leaky_relu'))

class VGG(nn.Module):
    def __init__(self, activation_func):
        super(VGG, self).__init__()
        actv_function = actfvs[activation_func]

        self.conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        layers = []
        for (num_convs, in_channels, num_channels) in self.conv_arch:
            layers += [vgg_block(num_convs, in_channels, num_channels, actv_function)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self, activation_func):
        super(LeNet5, self).__init__()
        actv_function = actfvs[activation_func]

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('actv1', actv_function()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('actv3', actv_function()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('actv5', actv_function())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('actv6', actv_function()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


def train(args, model, device, train_loader, optimizer, optimizer_activation, params_activation, epoch, losses):
    global cnt
    model.train()
    # l1_loss_target = torch.zeros_like(model.actv1.weight_denominator)
    # l2_denominator_crit = nn.L1Loss()
    train_loss = 0.
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(cnt)
        # if cnt == 13:
        #    print("break")
        # plot_activation(model, device, None, train_loader, epoch)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if optimizer_activation is not None:
            optimizer_activation.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # + 10*l2_denominator_crit(model.actv1.weight_denominator, l1_loss_target)
        loss.backward()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        clip_grad_norm_(model.parameters(), 5., norm_type=2)
        # clip_grad_value_(params_activation, .5)
        optimizer.step()
        if optimizer_activation is not None:
            optimizer_activation.step()

        # print statistics
        train_loss += loss.item()
        cnt += 1

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    return train_loss, train_acc

            # writer.add_scalar('train/loss', loss.item(), cnt)

            # print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
            #      epoch + 1, int(100 * (i + 1) / n_minibatches), running_loss / print_every,
            #      time.time() - start_time))

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #           100. * batch_idx / len(train_loader), loss.item()))
    # writer.add_scalar('train/loss_epoch', running_loss / len(train_loader.dataset), epoch)


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
    # writer.add_scalar('test/loss', test_loss, epoch)
    # writer.add_scalar('test/accuracy', acc, epoch)

    print('\nTest set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch, test_loss,
                                                                                            correct,
                                                                                            len(test_loader.dataset),
                                                                                            acc))
    return test_loss, acc


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--init', type=str, default="", choices=["", "xavier", "he"])

    args = parser.parse_args()
    scores = {}
    scores["train_loss"] = []
    scores["test_loss"] = []
    scores["train_acc"] = []
    scores["test_acc"] = []

    networks = dict({
        "vgg": VGG,
        "lenet": LeNet5,
    })

    network = networks[args.arch]

    global writer
    global cnt
    for activation_function_key in actfvs.keys():
        print("---" * 25)
        print("Starting with dataset: {}, activation function: {}".format(args.dataset, activation_function_key))
        print("---" * 25)
        # writer = SummaryWriter(comment=activation_function_key)
        save_path = 'examples/runs/mnist/paper_{}_{}_{}{}_seed{}/'.format(args.dataset, args.arch, args.optimizer,
                                                                          "_init_{}".format(args.init) if args.init != "" else "",
                                                             args.seed) + activation_function_key
        # writer = SummaryWriter(save_path)
        #
        # writer.add_scalar('configuration/batch size', args.batch_size)
        # writer.add_scalar('configuration/learning rate', args.lr)
        # writer.add_scalar('configuration/seed', args.seed)

        cnt = 0

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
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   transforms.RandomRotation(30),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
            lr_scheduler_milestones = [30, 60, 90]  # Simple CNN with 3 Conv
            # lr_scheduler_milestones = [40, 80]  # VGG
        elif args.dataset == 'fmnist':
            train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('../data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
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

        if args.init == 'xavier':
            model.apply(weights_init_xavier)
        elif args.init == 'he':
            model.apply(weights_init_he)

        params = list()
        params_activation = list()
        for p in model.named_parameters():
            if 'weight_center' in p[0] or 'weight_numerator' in p[0] or 'weight_denominator' in p[0]:
                if p[1].requires_grad:
                    params_activation.append(p[1])
            else:
                params.append(p[1])
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=500.)
        if args.optimizer.lower() == "adam":
            optimizer = optim.Adam(params, lr=args.lr)
        else:
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
        scheduler = None
        if "pade" in activation_function_key:
            if args.optimizer.lower() == "adam":
                optimizer_activation = optim.Adam(params_activation, lr=args.lr)
            else:
                optimizer_activation = optim.SGD(params_activation, lr=args.lr, momentum=args.momentum)
            scheduler_activation = None
        else:
            optimizer_activation = None
            scheduler_activation = None

        schedulers = list()
        if scheduler is not None:
            schedulers.append(scheduler)

        if scheduler_activation is not None:
            schedulers.append(scheduler_activation)

        losses = []

        # test(args, model, device, test_loader, 0)
        for epoch in range(1, args.epochs + 1):
            # if args.save_model:
            #     torch.save(model.state_dict(), os.path.join(save_path, "model_{}.pt".format(epoch)))

            train_loss, train_acc = train(args, model, device, train_loader, optimizer, optimizer_activation, None, epoch, losses)
            test_loss, test_acc = test(args, model, device, test_loader, epoch)
            scores["train_loss"].append(train_loss)
            scores["train_acc"].append(train_acc)
            scores["test_loss"].append(test_loss)
            scores["test_acc"].append(test_acc)
            # if epoch % 10 == 0:
            #     import ipdb; ipdb.set_trace()

            for current_scheduler in schedulers:
                current_scheduler.step()



        # writer.close()
        import pickle
        pickle.dump(scores, open(f"scores/scores_{args.arch}_{activation_function_key}_{args.seed}.pkl", "wb"))

        if args.save_model:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, f"model_final_{args.arch}_{activation_function_key}_{args.seed}.pt"))
            print(f"Saved in {save_path}")


if __name__ == '__main__':
    main()

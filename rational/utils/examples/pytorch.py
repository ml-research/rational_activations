import torchvision as vision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time


def evaluate_pytorch(model, test_loader, loss_function, device):
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        total = 0
        correct_pred = 0
        for index, data in enumerate(test_loader):
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.cuda(device), y_batch.cuda(device)
            
            pred = model(x_batch)
            loss = loss_function(pred, y_batch)
            valid_loss += loss.item()
            
            pred_classes = torch.max(pred.data, dim=1)[1]
            total += y_batch.size(0)
            correct_pred += (pred_classes == y_batch).sum().item()
            
    model.train()
    return correct_pred / total, valid_loss / len(test_loader)


def train_pytorch_model(model, train_loader, test_loader, device=0, epochs=40, vis_mod=10):
    model.cuda(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    loss_function = nn.CrossEntropyLoss()

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct_pred = 0
        for index, data in enumerate(train_loader):
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.cuda(device), y_batch.cuda(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_function(pred, y_batch)
            running_loss += loss.item()

            pred_classes = torch.max(pred.data, dim=1)[1]
            total += y_batch.size(0)
            correct_pred += (pred_classes == y_batch).sum().item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0)
            optimizer.step()

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct_pred / total)
        valid_acc, valid_loss = evaluate_pytorch(model, test_loader, loss_function, device)
        test_acc.append(valid_acc)
        test_loss.append(valid_loss)
        if (epoch + 1) % vis_mod == 0:
            print(f'[Epoch {epoch + 1:3d}] train_acc: {100 * correct_pred / total:5.2f}% - '\
                  f'train_loss: {running_loss / len(train_loader):6.3f}')
            print(f'[Epoch {epoch + 1:3d}] val_acc: {100 * valid_acc:5.2f}% - '\
                  f'val_loss: {valid_loss:6.3f}')
            print(f'[Epoch {epoch + 1:3d}] Model runtime: {time.time() - start:6.3f}s')
        
    return {'accuracy':train_acc, 'loss':train_loss,
            'val_accuracy':test_acc, 'val_loss':test_loss}


def prepare_data_pytorch(batch_size):
    train_transforms = vision.transforms.Compose([
        vision.transforms.Resize((32, 32)),
        vision.transforms.RandomRotation(30),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(0.1307, 0.3081),
    ])
    test_transforms = vision.transforms.Compose([
        vision.transforms.Resize((32, 32)),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(0.1307, 0.3081),
    ])

    mnist_train = vision.datasets.MNIST(root='./data/', download=True, train=True, transform=train_transforms)
    mnist_test = vision.datasets.MNIST(root='./data/', download=True, train=False, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=512, shuffle=False, num_workers=8)
    return train_loader, test_loader


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

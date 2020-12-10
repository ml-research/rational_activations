from mxnet import gluon, metric, autograd, gpu, cpu
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time


def prepare_data_mxnet(batch_size=256):
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.RandomRotation((-30, 30)),
        transforms.Normalize(0.1307, 0.3081),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ])

    train_data = gluon.data.DataLoader(
        datasets.MNIST(train=True).transform_first(train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=8)
    test_data = gluon.data.DataLoader(
        datasets.MNIST(train=False).transform_first(test_transforms),
        batch_size=batch_size, shuffle=False, num_workers=8)
    return train_data, test_data


def evaluate_mxnet(model, test_data, loss_fn, device):
    acc = metric.Accuracy()
    total_loss = 0.0
    for X, Y in test_data:
        X, Y = X.copyto(device), Y.copyto(device)
        pred = model(X)
        loss = loss_fn(pred, Y)
        
        total_loss += loss.mean().asscalar()
        acc.update(preds=pred, labels=Y)
    return acc.get()[1], total_loss / len(test_data)


def train_model_mxnet(model, train_data, test_data, device, epochs=40, vis_mod=10):
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 1e-2, 'momentum': 0.5, 'clip_gradient': 5.0})
    
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    start = time.time()
    for epoch in range(epochs):
        acc = metric.Accuracy()
        total_loss = 0.0

        for X, Y in train_data:
            X, Y = X.copyto(device), Y.copyto(device)
            with autograd.record():
                pred = model(X)
                loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step(batch_size=X.shape[0])
            total_loss += loss.mean().asscalar()

            acc.update(preds=pred, labels=Y)

        train_loss.append(total_loss / len(train_data))
        train_acc.append(acc.get()[1])
        _test_acc, _test_loss = evaluate_mxnet(model, test_data, loss_fn, device)
        test_acc.append(_test_acc)
        test_loss.append(_test_loss)
        if (epoch + 1) % vis_mod == 0:
            print(f'[Epoch {epoch + 1:3d}] train_acc: {100 * acc.get()[1]:4.2f}% - '\
                  f'train_loss: {total_loss / len(train_data):6.3f}')
            print(f'[Epoch {epoch + 1:3d}] val_acc: {100 * _test_acc:5.2f}% - '\
                  f'val_loss: {_test_loss:6.3f}')

            print(f'Model runtime: {time.time() - start:6.3f}s')
    
    return {'accuracy':train_acc, 'loss':train_loss,
            'val_accuracy':test_acc, 'val_loss':test_loss}

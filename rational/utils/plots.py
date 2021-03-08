import matplotlib.pyplot as plt
from rational.torch import Rational
import math
import numpy as np
import torch


def plot(data, title, figsize=(10, 4), ylim=(0, 1), legend=True):
    plt.figure(figsize=figsize)
    for label, points in data.items():
        plt.plot(range(len(points)), points, label=label)
    plt.title(title)
    plt.ylim(ylim)
    if legend:
        plt.legend()
    plt.show()


def plot_grid(data, train_acc_ylim=(0.95, 1.01), train_loss_ylim=(-0.05, 0.1), valid_acc_ylim=(0.95, 1.01),
              valid_loss_ylim=(-0.05, 0.1), legend=True, size=(14, 8), linewidth=2, title='Training and Validation Metrics'):
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.03)
    axs = gs.subplots(sharex='col')
    (ax1, ax2), (ax3, ax4) = axs
    ax11, ax22 = axs
    fig.suptitle(title)

    for key in data:
        ax1.plot(range(len(data[key]['accuracy'])), data[key]['accuracy'], label=key, linewidth=linewidth)
        #ax2.plot(range(len(data[key]['val_accuracy'])), data[key]['val_accuracy'], label=key, linewidth=linewidth)
        ax3.plot(range(len(data[key]['loss'])), data[key]['loss'], label=key, linewidth=linewidth)
        #ax4.plot(range(len(data[key]['val_loss'])), data[key]['val_loss'], label=key, linewidth=linewidth)
        
        ax2.plot(range(len(data[key]['test_accuracy'])), data[key]['test_accuracy'], label=key, linewidth=linewidth)
        ax4.plot(range(len(data[key]['test_loss'])), data[key]['test_loss'], label=key, linewidth=linewidth)

    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(train_acc_ylim)
    ax2.yaxis.tick_right()
    ax2.set_ylim(valid_acc_ylim)
    ax3.set_xlabel('Training')
    ax3.set_ylabel('Loss')
    ax3.set_ylim(train_loss_ylim)
    ax4.set_xlabel('Test')
    #ax4.set_xlabel('Validation')
    ax4.set_ylim(valid_loss_ylim)
    ax4.yaxis.tick_right()
    
    #for ax in axs.flat:
    #    ax.label_outer()
    if legend:
        plt.legend()
    plt.show()


def plot_rational_functions(model, input_range=None, include_custom_func=None):
    rationals = _collect_rational_functions(model, rationals=[])
    plots = []
    for act in rationals:
        fig = act.show(display=False, input_range=input_range)
        if include_custom_func is not None and input_range is not None:
                axes = fig.axes[0]
                points = include_custom_func(input_range.cpu()).numpy()
                axes.plot(input_range.cpu().numpy(), points, color='red')
        plt.show()


def _collect_rational_functions(model, rationals=[]):
    for n, c in model.named_children():
        if isinstance(c, Rational):
            rationals.append(c)
        else:
            _collect_rational_functions(c, rationals=rationals)
    return rationals


def activate_input_retrieve_mode(model, auto_stop=True, max_saves=1000, bin_width=0.1):
    for n, c in model.named_children():
        if isinstance(c, Rational):
            c.input_retrieve_mode(auto_stop=auto_stop, max_saves=max_saves, bin_width=bin_width)
        else:
            activate_input_retrieve_mode(model=c, auto_stop=auto_stop, max_saves=max_saves, bin_width=bin_width)

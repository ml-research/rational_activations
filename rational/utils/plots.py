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


def plot_grid(data, acc_ylim=(0.95, 1), loss_ylim=(0, 0.1), legend=True, size=(14, 8)):
    x = range(len(data['relu']['accuracy']))
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.03)
    axs = gs.subplots(sharex='col', sharey='row')
    (ax1, ax2), (ax3, ax4) = axs
    ax11, ax22 = axs
    fig.suptitle('Training and Validation Metrics')

    if 'rational' in data:
        ax1.plot(x, data['rational']['accuracy'], label='rational')
        ax2.plot(x, data['rational']['val_accuracy'], label='rational')
        ax3.plot(x, data['rational']['loss'], label='rational')
        ax4.plot(x, data['rational']['val_loss'], label='rational')

    if 'relu' in data:
        ax1.plot(x, data['relu']['accuracy'], label='relu')
        ax2.plot(x, data['relu']['val_accuracy'], label='relu')
        ax3.plot(x, data['relu']['loss'], label='relu')
        ax4.plot(x, data['relu']['val_loss'], label='relu')

    if 'mish' in data:
        ax1.plot(x, data['mish']['accuracy'], label='mish')
        ax2.plot(x, data['mish']['val_accuracy'], label='mish')
        ax3.plot(x, data['mish']['loss'], label='mish')
        ax4.plot(x, data['mish']['val_loss'], label='mish')

    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(acc_ylim)
    ax2.set_ylim(acc_ylim)
    ax3.set_xlabel('Training')
    ax3.set_ylabel('Loss')
    ax3.set_ylim(loss_ylim)
    ax4.set_xlabel('Validation')
    ax4.set_ylim(loss_ylim)
    
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.show()


def plot_rational_functions(model, include_custom_func=None):
    rationals = _collect_rational_functions(model, rationals=[])
    plots = []
    for act in rationals:
        fig = act.show(display=False, input_range=None)
        if include_custom_func is not None:
                axes = fig.axes[0]
                ylim = axes.get_ylim()
                input_range = np.arange(ylim[0], ylim[1], 1)
                points = include_custom_func(torch.tensor(input_range).float()).numpy()
                axes.plot(input_range, points, color='red')
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

import matplotlib.pyplot as plt


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
    ax1.plot(x, data['rational']['accuracy'], label='rational')
    ax1.plot(x, data['relu']['accuracy'], label='relu')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(acc_ylim)
    
    ax2.plot(x, data['rational']['val_accuracy'], label='rational')
    ax2.plot(x, data['relu']['val_accuracy'], label='relu')
    ax2.set_ylim(acc_ylim)
    
    ax3.plot(x, data['rational']['loss'], label='rational')
    ax3.plot(x, data['relu']['loss'], label='relu')
    ax3.set_xlabel('Training')
    ax3.set_ylabel('Loss')
    ax3.set_ylim(loss_ylim)
    
    ax4.plot(x, data['rational']['val_loss'], label='rational')
    ax4.plot(x, data['relu']['val_loss'], label='relu')
    ax4.set_xlabel('Validation')
    ax4.set_ylim(loss_ylim)

    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.show()
    

def plot_rational_functions(model):
    pass

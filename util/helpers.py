import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import *
from torch.autograd import Variable


############################################################################
# Helper Utilities
############################################################################


def weights_init_normal(m):
    # Set initial state of weights
    classname = m.__class__.__name__
    if 'ConvTrans' == classname:
        pass
    elif 'Conv2d' in classname or 'Linear' in classname or 'ConvTrans' in classname:
        nn.init.normal(m.weight.data, 0, .02)


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))


############################################################################
# Display Images
############################################################################


def show_test(params, denorm, dataloader, model, save=False):
    # Show and save
    ids_a = params['ids']
    image_grid_len = len(ids_a)
    fig, ax = plt.subplots(image_grid_len, 3, figsize=(10, 4*image_grid_len))
    count = 0
    model.eval()
    for idx, real in enumerate(dataloader):
        if idx in ids_a:
            real_a = Variable(real[0].cuda())
            real_b = Variable(real[1].cuda())
            test = model(real_a)
            ax[count, 0].cla()
            ax[count, 0].imshow(denorm.denorm(real_a[0]))
            ax[count, 1].cla()
            ax[count, 1].imshow(denorm.denorm(real_b[0]))
            ax[count, 2].cla()
            ax[count, 2].imshow(denorm.denorm(test[0]))
            count += 1
    model.train()
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)

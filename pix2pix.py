import math
import torch.optim as optim
import torch
import time
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn

from util import helpers as helper
from util import loaders as load
from models import networks as n


############################################################################
# Train
############################################################################

class Pix2Pix:
    """
    Example usage if not using command line:

     params = {'dataset': 'edges2shoes',
              'train_folder': 'train',
              'test_folder': 'val',
              'in_channels': 3,
              'batch_size': 16,
              'gen_filters': 1024,
              'disc_filters': 512,
              'lr_disc': 1e-4,
              'lr_gen': 1e-4,
              'test_perc': .01,
              'lr_cycle_mult': 2,
              'beta1': .5,
              'beta2': .999,
              'train_epoch': 2,
              'alpha' : 10,
              'gen_layers': 6,
              'disc_layers': 4,
              'img_output_size':256,
              'ids': [0, 1],
              'save_root': 'shoes'}
    p2p = Pix2Pix(params)
    p2p.train()
    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0
        self.current_cycle = 0

        # Setup data loaders
        self.transform = load.NormDenorm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.train_loader, data_len = load.data_load(f'/data/{params["dataset"]}/{params["train_folder"]}/',
                                                     self.transform,
                                                     params["batch_size"],
                                                     shuffle=True,
                                                     output_res=params["img_output_size"],
                                                     perc=params["test_perc"])

        self.test_loader, test_data_len = load.data_load(f'/data/{params["dataset"]}/{params["test_folder"]}/',
                                                         self.transform,
                                                         1,
                                                         shuffle=False,
                                                         perc=params["test_perc"],
                                                         output_res=params["img_output_size"],
                                                         train=False)
        # Set learning rate schedule
        self.set_lr_sched(params['train_epoch'],
                          math.ceil(float(data_len) / float(params['batch_size'])),
                          params['lr_cycle_mult'])

        # Setup models
        self.model_dict["G"] = n.Generator(layers=params["gen_layers"],
                                           filts=params["gen_filters"],
                                           channels=params["in_channels"])

        self.model_dict["D"] = n.Discriminator(layers=params["disc_layers"],
                                               filts=params["disc_filters"],
                                               channels=params["in_channels"] * 2)

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')

        # Setup losses
        self.BCE_loss = nn.BCELoss()
        self.L1_loss = nn.L1Loss()

        # Setup optimizers
        self.opt_dict["G"] = optim.Adam(self.model_dict["G"].parameters(),
                                        lr=params['lr_gen'],
                                        betas=(params['beta1'], params['beta2']),
                                        weight_decay=.00001)

        self.opt_dict["D"] = optim.Adam(self.model_dict["D"].parameters(),
                                        lr=params['lr_disc'],
                                        betas=(params['beta1'], params['beta2']),
                                        weight_decay=.00001)

        print('Losses Initialized')

        # Setup history storage
        self.losses = ['D_loss', 'G_D_loss', 'G_L_loss']
        self.loss_batch_dict = {}
        self.loss_batch_dict_test = {}
        self.loss_epoch_dict = {}
        self.loss_epoch_dict_test = {}
        self.train_hist_dict = {}
        self.train_hist_dict_test = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []
            self.train_hist_dict_test[loss] = []
            self.loss_epoch_dict_test[loss] = []
            self.loss_batch_dict_test[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1
        self.current_cycle = state['cycle'] + 1

        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])

        self.train_hist_dict = state['train_hist']
        self.train_hist_dict_test = state['train_hist_test']

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()

        model_state = {'iter': self.current_iter,
                       'cycle': self.current_cycle,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict,
                       'train_hist_test': self.train_hist_dict_test
                       }

        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def lr_lookup(self):
        # Determine proper learning rate multiplier for this iter
        lr_mult = self.iter_stack[self.current_iter]
        save = self.current_iter in self.save_index
        return lr_mult, save

    def set_lr_sched(self, epochs, iters, mult):
        # Test implementation of warm restarts
        mult_iter = iters
        iter_stack = []
        save_index = []
        for a in range(epochs):
            iter_stack += [math.cos((x / mult_iter) * 3.14) * .5 + .5 for x in (range(int(mult_iter)))]
            mult_iter *= mult
            save_index.append(len(iter_stack) - 1)

        self.iter_stack = iter_stack
        self.save_index = save_index

        fig = plt.figure()
        plt.plot(self.iter_stack)
        plt.savefig(f'output/{self.params["save_root"]}_learning_rate_schedule.jpg')
        plt.ylabel('Learning Rate Schedule')
        plt.show()
        plt.close(fig)

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            x_test = range(len(self.train_hist_dict_test[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
                plt.plot(x_test, self.train_hist_dict_test[key], label=key + '_test')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def set_grad_req(self, d=True, g=True):
        # Easily enable and disable gradient storage per model
        for par in self.model_dict["D"].parameters():
            par.requires_grad = d
        for par in self.model_dict["G"].parameters():
            par.requires_grad = g

    def train_gen(self, real_a, real_b, fake_b):
        # train function for generator
        self.set_grad_req(d=False, g=True)
        self.opt_dict["G"].zero_grad()

        # concat real_a and fake_b then discriminate
        fake_cat = torch.cat([real_a, fake_b], 1)
        d_result = self.model_dict["D"](fake_cat)
        self.loss_batch_dict['G_D_loss'] = self.BCE_loss(d_result, Variable(torch.ones(d_result.size()).cuda()))
        # L1 loss
        self.loss_batch_dict['G_L_loss'] = self.L1_loss(fake_b, real_b) * self.params['alpha']

        # addup generator loss and step
        g_loss_total = self.loss_batch_dict['G_D_loss'] + (self.loss_batch_dict['G_L_loss'])
        g_loss_total.backward()
        self.opt_dict["G"].step()

    def train_disc(self, real_a, real_b, fake_b):
        # train function for discriminator
        self.set_grad_req(d=True, g=False)
        self.opt_dict["D"].zero_grad()

        # concat real_a and fake_b, real_a and real_b
        fake_cat = torch.cat([real_a.detach(), fake_b.detach()], 1)
        real_cat = torch.cat([real_a, real_b], 1)

        # train discriminator
        d_real = self.model_dict["D"](real_cat)
        d_real_loss = self.BCE_loss(d_real, Variable(torch.ones(d_real.size()).cuda()))
        d_fake = self.model_dict["D"](fake_cat)
        d_fake_loss = self.BCE_loss(d_fake, Variable(torch.zeros(d_fake.size()).cuda()))

        # add up disc a loss and step
        self.loss_batch_dict['D_loss'] = (d_real_loss + d_fake_loss) * .5
        self.loss_batch_dict['D_loss'].backward()
        self.opt_dict["D"].step()

    def test_gen(self, real_a, real_b, fake_b):
        # test function for generator

        # concat real_a and fake_b then discriminate
        fake_cat = torch.cat([real_a, fake_b], 1)
        d_result = self.model_dict["D"](fake_cat)
        self.loss_batch_dict_test['G_D_loss'] = self.BCE_loss(d_result, Variable(torch.ones(d_result.size()).cuda()))
        # L1 loss
        self.loss_batch_dict_test['G_L_loss'] = self.L1_loss(fake_b, real_b) * self.params['alpha']

    def test_disc(self, real_a, real_b, fake_b):
        # test function for discriminator

        # concat real_a and fake_b, real_a and real_b
        fake_cat = torch.cat([real_a.detach(), fake_b.detach()], 1)
        real_cat = torch.cat([real_a, real_b], 1)

        # test dicriminator
        d_real = self.model_dict["D"](real_cat)
        d_real_loss = self.BCE_loss(d_real, Variable(torch.ones(d_real.size()).cuda()))
        d_fake = self.model_dict["D"](fake_cat)
        d_fake_loss = self.BCE_loss(d_fake, Variable(torch.zeros(d_fake.size()).cuda()))

        # add up disc a loss
        self.loss_batch_dict_test['D_loss'] = (d_real_loss + d_fake_loss) * .5

    def test_loop(self):
        # Test on validation set
        self.model_dict["D"].eval()
        self.model_dict["G"].eval()
        self.opt_dict["G"].zero_grad()
        self.opt_dict["D"].zero_grad()

        for loss in self.losses:
            self.loss_epoch_dict_test[loss] = []
        self.set_grad_req(d=False, g=False)
        # test loop #
        for (real_a, real_b) in tqdm(self.test_loader):
            real_a, real_b = Variable(real_a.cuda()), Variable(real_b.cuda())
            # GENERATE
            fake_b = self.model_dict["G"](real_a)
            # TEST DISCRIMINATOR
            self.test_disc(real_a, real_b, fake_b)
            # TEST GENERATOR
            self.test_gen(real_a, real_b, fake_b)
            # append all losses in loss dict #
            [self.loss_epoch_dict_test[loss].append(self.loss_batch_dict_test[loss].data[0]) for loss in self.losses]

        [self.train_hist_dict_test[loss].append(helper.mft(self.loss_epoch_dict_test[loss])) for loss in self.losses]

    def train(self):
        # Train following learning rate schedule
        params = self.params
        done = False
        while not done:
            # clear last epochs losses
            for loss in self.losses:
                self.loss_epoch_dict[loss] = []

            self.model_dict["D"].train()
            self.model_dict["G"].train()
            self.set_grad_req(d=True, g=True)

            epoch_start_time = time.time()
            num_iter = 0

            print(f"Sched Cycle:{self.current_cycle}, Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
            [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in
             self.opt_dict.keys()]

            for (real_a, real_b) in tqdm(self.train_loader):

                if self.current_iter > len(self.iter_stack) - 1:
                    done = True
                    self.display_history()
                    print('Hit End of Learning Schedule!')
                    break

                # set learning rate
                lr_mult, save = self.lr_lookup()
                self.opt_dict["D"].param_groups[0]['lr'] = lr_mult * params['lr_disc']
                self.opt_dict["G"].param_groups[0]['lr'] = lr_mult * params['lr_gen']

                real_a, real_b = Variable(real_a.cuda()), Variable(real_b.cuda())

                # GENERATE
                fake_b = self.model_dict["G"](real_a)
                # TRAIN DISCRIMINATOR
                self.train_disc(real_a, real_b, fake_b)
                # TRAIN GENERATOR
                self.train_gen(real_a, real_b, fake_b)

                # append all losses in loss dict
                [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].data[0]) for loss in self.losses]

                if save:
                    save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                    tqdm.write(save_str)
                    self.current_epoch += 1

                self.current_iter += 1
                num_iter += 1

            # generate test images and save to disk
            helper.show_test(params,
                             self.transform,
                             self.test_loader,
                             self.model_dict['G'],
                             save=f'output/{params["save_root"]}_{self.current_cycle}.jpg')

            # run validation set loop to get losses
            self.test_loop()

            if not done:
                self.current_cycle += 1
                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time
                print(f'Epoch Training Training Time: {per_epoch_ptime}')
                [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
                [print(f'Val {loss}: {helper.mft(self.loss_epoch_dict_test[loss])}') for loss in self.losses]
                print('\n')
                [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]

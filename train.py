#!/usr/bin/env python
import argparse
from pix2pix import Pix2Pix


parser = argparse.ArgumentParser()

parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset', nargs='?', default='edges2shoes', type=str)
parser.add_argument('--train_folder', nargs='?', default='train', type=str)
parser.add_argument('--test_folder', nargs='?', default='val', type=str)
parser.add_argument('--in_channels', nargs='?', default=3, type=int)
parser.add_argument('--batch_size', nargs='?', default=16, type=int)
parser.add_argument('--gen_filters', nargs='?', default=1024, type=int)
parser.add_argument('--disc_filters', nargs='?', default=512, type=int)
parser.add_argument('--img_output_size', nargs='?', default=256, type=int)
parser.add_argument('--gen_layers', nargs='?', default=6, type=int)
parser.add_argument('--disc_layers', nargs='?', default=4, type=int)
parser.add_argument('--test_perc', nargs='?', default=1, type=float)
parser.add_argument('--lr_disc', nargs='?', default=1e-4, type=float)
parser.add_argument('--lr_gen', nargs='?', default=1e-4, type=float)
parser.add_argument('--lr_cycle_mult', nargs='?', default=2, type=int)
parser.add_argument('--beta1', nargs='?', default=.5, type=float)
parser.add_argument('--beta2', nargs='?', default=.999, type=float)
parser.add_argument('--alpha', nargs='?', default=10, type=int)
parser.add_argument('--train_epoch', nargs='?', default=6, type=int)
parser.add_argument('--ids', type=int, nargs='+', default=[1, 53, 62])
parser.add_argument('--save_root', nargs='?', default='shoes', type=str)
parser.add_argument('--load_state', nargs='?', type=str)

params = vars(parser.parse_args())

# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    p2p = Pix2Pix(params)
    if params['load_state']:
        p2p.load_state(params['load_state'])
    else:
        print('Starting From Scratch')
    p2p.train()

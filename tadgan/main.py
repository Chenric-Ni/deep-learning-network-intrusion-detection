import os
import argparse

from torch.backends import cudnn
from utils import *

from solve import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--win_size', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=41)
    parser.add_argument('--output_dim', type=int, default=41)
    parser.add_argument('--hidden_dim', type=list, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='KDD')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--model_name', type=str, default='tadagan', choices=['baseline', 'tadagan'])
    parser.add_argument('--data_path', type=str, default=r'./data/NSL-KDD')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints')


    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)

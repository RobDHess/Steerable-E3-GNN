import torch
import argparse
import os
import numpy as np
import torch.multiprocessing as mp
from e3nn.o3 import Irreps, spherical_harmonics
from models.balanced_irreps import BalancedIrreps, WeightBalancedIrreps


def _find_free_port():
    """ Find free port, so multiple runs don't clash """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=False,
                        help='logging flag')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="qm9",
                        help='Data set')
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=False,
                        help='Download flag')

    # QM9 parameters
    parser.add_argument('--target', type=str, default="alpha",
                        help='Target value, also used for gravity dataset [pos, force]')
    parser.add_argument('--radius', type=float, default=2,
                        help='Radius (Angstrom) between which atoms to add links.')
    parser.add_argument('--feature_type', type=str, default="one_hot",
                        help='Type of input feature: one-hot, or Cormorants charge thingy')

    # Nbody parameters:
    parser.add_argument('--nbody_name', type=str, default="nbody_small",
                        help='Name of nbody data [nbody, nbody_small]')
    parser.add_argument('--max_samples', type=int, default=3000,
                        help='Maximum number of samples in nbody dataset')
    parser.add_argument('--time_exp', type=bool, default=False,
                        help='Flag for timing experiment')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')

    # Gravity parameters:
    parser.add_argument('--neighbours', type=int, default=6,
                        help='Number of connected nearest neighbours')

    # Model parameters
    parser.add_argument('--model', type=str, default="segnn",
                        help='Model name')
    parser.add_argument('--hidden_features', type=int, default=128,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_h', type=int, default=2,
                        help='max degree of hidden rep')
    parser.add_argument('--lmax_attr', type=int, default=3,
                        help='max degree of geometric attribute embedding')
    parser.add_argument('--subspace_type', type=str, default="weightbalanced",
                        help='How to divide spherical harmonic subspaces')
    parser.add_argument('--layers', type=int, default=7,
                        help='Number of message passing layers')
    parser.add_argument('--norm', type=str, default="instance",
                        help='Normalisation type [instance, batch]')
    parser.add_argument('--pool', type=str, default="avg",
                        help='Pooling type type [avg, sum]')
    parser.add_argument('--conv_type', type=str, default="linear",
                        help='Linear or non-linear aggregation of local information in SEConv')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=0, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    args = parser.parse_args()

    # Select dataset.
    if args.dataset == "qm9":
        from qm9.train import train
        task = "graph"
        if args.feature_type == "one_hot":
            input_irreps = Irreps("5x0e")
        elif args.feature_type == "cormorant":
            input_irreps = Irreps("15x0e")
        elif args.feature_type == "gilmer":
            input_irreps = Irreps("11x0e")
        output_irreps = Irreps("1x0e")

        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("1x0e")
    elif args.dataset == "nbody":
        from nbody.train_nbody import train
        task = "node"
        input_irreps = Irreps("2x1o + 1x0e")
        output_irreps = Irreps("1x1o")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("2x0e")
    elif args.dataset == "gravity":
        from nbody.train_gravity import train
        task = "node"
        input_irreps = Irreps("2x1o + 1x0e")
        output_irreps = Irreps("1x1o")
        edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
        additional_message_irreps = Irreps("2x0e")
    else:
        raise Exception("Dataset could not be found")

    # Create hidden irreps
    if args.subspace_type == "weightbalanced":
        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(args.hidden_features)), node_attr_irreps, sh=True, lmax=args.lmax_h)
    elif args.subspace_type == "balanced":
        hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
    else:
        raise Exception("Subspace type not found")

    # Select model
    if args.model == "segnn":
        from models.segnn.segnn import SEGNN
        model = SEGNN(input_irreps,
                      hidden_irreps,
                      output_irreps,
                      edge_attr_irreps,
                      node_attr_irreps,
                      num_layers=args.layers,
                      norm=args.norm,
                      pool=args.pool,
                      task=task,
                      additional_message_irreps=additional_message_irreps)
        args.ID = "_".join([args.model, args.dataset, args.target, str(np.random.randint(1e4, 1e5))])
    elif args.model == "seconv":
        from models.segnn.seconv import SEConv
        model = SEConv(input_irreps,
                       hidden_irreps,
                       output_irreps,
                       edge_attr_irreps,
                       node_attr_irreps,
                       num_layers=args.layers,
                       norm=args.norm,
                       pool=args.pool,
                       task=task,
                       additional_message_irreps=additional_message_irreps,
                       conv_type=args.conv_type)
        args.ID = "_".join([args.model, args.conv_type, args.dataset, str(np.random.randint(1e4, 1e5))])
    else:
        raise Exception("Model could not be found")

    print(model)
    print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))
    if args.gpus == 0:
        print('Starting training on the cpu...')
        args.mode = 'cpu'
        train(0, model, args)
    elif args.gpus == 1:
        print('Starting training on a single gpu...')
        args.mode = 'gpu'
        train(0, model, args)
    elif args.gpus > 1:
        print('Starting training on', args.gpus, 'gpus...')
        args.mode = 'gpu'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        port = _find_free_port()
        print('found free port', port)
        os.environ['MASTER_PORT'] = str(port)
        mp.spawn(train, nprocs=args.gpus, args=(model, args,))

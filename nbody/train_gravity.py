import argparse
import numpy as np
import torch
import os
import wandb
from torch import nn, optim
import time
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from e3nn.o3 import Irreps, spherical_harmonics
from torch_scatter import scatter
from torch_geometric.nn import knn_graph


from nbody.dataset_gravity import GravityDataset

time_exp_dic = {'time': 0, 'counter': 0}


class O3Transform:
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph):
        pos = graph.pos
        vel = graph.vel
        mass = graph.mass

        prod_mass = mass[graph.edge_index[0]] * mass[graph.edge_index[1]]
        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce="mean") + vel_embedding

        vel_abs = torch.sqrt(vel.pow(2).sum(1, keepdims=True))
        mean_pos = pos.mean(1, keepdims=True)

        graph.x = torch.cat((pos - mean_pos, vel, vel_abs), 1)
        graph.additional_message_features = torch.cat((edge_dist, prod_mass), dim=-1)
        return graph


def train(gpu, model, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = torch.device('cuda:' + str(gpu))

    dataset_train = GravityDataset(partition='train', dataset_name=args.nbody_name,
                                   max_samples=args.max_samples, neighbours=args.neighbours, target=args.target)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = GravityDataset(partition='val', dataset_name=args.nbody_name,
                                 neighbours=args.neighbours, target=args.target)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = GravityDataset(partition='test', dataset_name=args.nbody_name,
                                  neighbours=args.neighbours, target=args.target)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()
    transform = O3Transform(args.lmax_attr)

    if args.log and gpu == 0:
        if args.time_exp:
            wandb.init(project="Gravity time", name=args.ID, config=args, entity="segnn")
        else:
            wandb.init(project="SEGNN Gravity", name=args.ID, config=args, entity="segnn")

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_train, transform, device, args)
        if args.log and gpu == 0:
            wandb.log({"Train MSE": train_loss})
        if epoch % args.test_interval == 0 or epoch == args.epochs-1:
            #train(epoch, loader_train, backprop=False)
            val_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_val, transform, device, args, backprop=False)
            test_loss = run_epoch(model, optimizer, loss_mse, epoch, loader_test,
                                  transform, device, args, backprop=False)
            if args.log and gpu == 0:
                wandb.log({"Val MSE": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))

    if args.log and gpu == 0:
        wandb.log({"Test MSE": best_test_loss})
    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, transform, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, force, mass, y = data

        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        if args.model == 'segnn' or args.model == 'seconv':
            graph = Data(pos=loc, vel=vel, force=force, mass=mass, y=y)
            batch = torch.arange(0, batch_size)
            graph.batch = batch.repeat_interleave(n_nodes).long()
            graph.edge_index = knn_graph(loc, args.neighbours, graph.batch)

            graph = transform(graph)  # Add O3 attributes
            graph = graph.to(device)
            pred = model(graph)
        else:
            raise Exception("Wrong model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            if epoch % 100 == 99:
                print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
                if args.log:
                    wandb.log({"Time": time_exp_dic['time']/time_exp_dic['counter']})
        loss = criterion(pred, graph.y)
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']

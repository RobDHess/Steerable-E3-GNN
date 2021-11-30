import wandb
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

from qm9.dataset import QM9
from qm9.evaluate import evaluate
import utils


def train(gpu, model, args):
    if args.gpus == 0:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
        if args.gpus > 1:
            dist.init_process_group("nccl", rank=gpu, world_size=args.gpus)
            torch.cuda.set_device(gpu)

    model = model.to(device)
    if args.gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    # Create datasets and dataloaders
    train_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "train", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu)
    valid_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "valid", args.lmax_attr,
                                             feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)
    test_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "test", args.lmax_attr,
                                            feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu, train=False)

    # Get train set statistics
    target_mean, target_mad = train_loader.dataset.calc_stats()

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.8*(args.epochs)), int(0.9*(args.epochs))], verbose=True)
    criterion = nn.L1Loss()

    # Logging parameters
    target = args.target
    best_valid_MAE = 1e30
    i = 0
    N_samples = 0
    loss_sum = 0
    train_MAE_sum = 0

    # Init wandb
    if args.log and gpu == 0:
        wandb.init(project="SEGNN " + args.dataset + " " + args.target, name=args.ID, config=args, entity="segnn")

    # Let's start!
    if gpu == 0:
        print("Training:", args.ID)
    for epoch in range(args.epochs):
        # Set epoch so shuffling works right in distributed mode.
        if args.gpus > 1:
            train_loader.sampler.set_epoch(epoch)
        # Training loop

        for step, graph in enumerate(train_loader):
            # Forward & Backward.
            graph = graph.to(device)
            out = model(graph).squeeze()
            loss = criterion(out, (graph.y - target_mean)/target_mad)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            i += 1
            N_samples += graph.y.size(0)
            loss_sum += loss
            train_MAE_sum += criterion(out.detach()*target_mad + target_mean, graph.y)*graph.y.size(0)

            # Report
            if i % args.print == 0:
                print("epoch:%2d  step:%4d  loss: %0.4f  train MAE:%0.4f" %
                      (epoch, step, loss_sum/i, train_MAE_sum/N_samples))

                if args.log and gpu == 0:
                    wandb.log({"loss": loss_sum/i, target + " train MAE": train_MAE_sum /
                               N_samples})

                i = 0
                N_samples = 0
                loss_sum = 0
                train_MAE_sum = 0

        # Evaluate on validation set
        valid_MAE = evaluate(model, valid_loader, criterion, device, args.gpus, target_mean, target_mad)
        # Save best validation model
        if valid_MAE < best_valid_MAE:
            best_valid_MAE = valid_MAE
            utils.save_model(model, args.save_dir, args.ID, device)
        if gpu == 0:
            print("VALIDATION: epoch:%2d  step:%4d  %s-MAE:%0.4f" %
                  (epoch, step, target, valid_MAE))
            if args.log:
                wandb.log({target + " val MAE": valid_MAE})

        # Adapt learning rate
        scheduler.step()

    # Final evaluation on test set
    model = utils.load_model(model, args.save_dir, args.ID, device)
    test_MAE = evaluate(model, test_loader, criterion, device, args.gpus, target_mean, target_mad, )
    if gpu == 0:
        print("TEST: epoch:%2d  step:%4d  %s-MAE:%0.4f" %
              (epoch, step, target, test_MAE))
        if args.log:
            wandb.log({target + " test MAE": test_MAE})
            wandb.save(os.path.join(args.save_dir, args.ID + "_" + device + ".pt"))

    if args.log and gpu == 0:
        wandb.finish()
    if args.gpus > 1:
        dist.destroy_process_group()

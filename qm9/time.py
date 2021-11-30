import time
import wandb
from tqdm import tqdm
import torch

from models.segnn.segnn import SEGNN
from qm9.dataset import QM9
import utils

time_dict = {'time': 0, 'counter': 0}


def main(gpu, model, args):
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

    dataloader = train_loader = utils.make_dataloader(QM9(args.root, args.target, args.radius, "train", args.lmax_attr,
                                                          feature_type=args.feature_type), args.batch_size, args.num_workers, args.gpus, gpu)

    if args.log:
        wandb.init(project="SEGNN " + args.dataset + " " + "time", name=args.ID, config=args, entity="segnn")

    n = 0
    with tqdm(total=args.forward_passes + args.warmup) as pbar:
        while n < args.forward_passes + args.warmup:
            for i, graph in enumerate(dataloader):
                graph = graph.to(device)
                if n > args.warmup:
                    if device != "cpu":
                        torch.cuda.synchronize()
                    t1 = time.time()

                out = model(graph)  # Forward pass

                if n > args.warmup:
                    if device != "cpu":
                        torch.cuda.synchronize()
                    t2 = time.time()
                    time_dict['time'] += t2 - t1
                    time_dict['counter'] += 1

                pbar.update()
                n += 1
                if n == args.forward_passes + args.warmup:
                    break

    T = time_dict['time']/time_dict['counter']
    if args.log and gpu == 0:
        wandb.log({"time": T})

    print("Forward pass time is", T, "over", args.forward_passes, "forward passes with batch size", args.batch_size)

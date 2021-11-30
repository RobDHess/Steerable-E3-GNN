import torch
import torch.distributed as dist


def evaluate(model, dataloader, criterion, device, world_size, loc=0, scale=1):
    """ Evaluate a model on a specific dataloader, with distributed communication (if necessary) """
    model.eval()
    N = torch.zeros(1).to(device)
    score = torch.zeros(1).to(device)

    with torch.no_grad():
        for graph in dataloader:
            graph = graph.to(device)
            out = model(graph).squeeze()

            n = graph.y.size(0)
            N += n
            score += n*criterion(out*scale + loc, graph.y)

    model.train()
    if world_size > 1:
        dist.all_reduce(score)
        dist.all_reduce(N)

    return (score/N).item()

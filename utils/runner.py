import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def get_shapes(cfg, nlabs):
    """
    Define a collection of shapes that makes reference prior calculation easy
    """
    shapes = []
    bs = (cfg.ref.μ * cfg.hp.bs) // cfg.ref.order
    shp = [bs, cfg.ref.particles] + ([1] * (cfg.ref.order))
    for i in range(cfg.ref.order):
        shp[2+i] = nlabs
        if i != 0:
            shp[1+i] = 1
        shapes.append(list(shp))
    return shapes


def log_metrics(epoch, net, test_loader):
    ret = evaluate_model(net, test_loader)
    ret_ema = evaluate_model(net, test_loader, True)
    info = {'epoch': epoch,
            'test_acc': ret[0],
            'test_loss': ret[1],
            'ema_test_acc': ret_ema[0],
            'ema_test_loss': ret_ema[1],
            'particle_accs': list(ret[2])}
    print(info)


def evaluate_model(net, dataloader, use_ema=False):
    """
    Evaluate the network
    """
    acc = 0.0
    loss = 0.0
    count = 0.0
    acc2 = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    net.eval()
    with torch.inference_mode():
        for dat, labels in dataloader:
            batch_size = int(labels.size()[0])
            labels = labels.long()

            dat = dat.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            out = net(dat, use_ema)
            ensemble = torch.mean(out, axis=2)
            loss += (criterion(ensemble, labels).item())

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            ensemble = ensemble.cpu().detach().numpy()
            acc += np.sum(labels == (np.argmax(ensemble, axis=1)))
            acc2 += np.sum(labels[:, None] == (np.argmax(out, axis=1)), axis=0)

            count += batch_size

    ret = (np.round((acc/count) * 100, 2),
           np.round(loss/count, 3),
           np.round((acc2/count) * 100, 3))
    return ret


def get_minibatch(iters, loaders, particles):

    # Fetch labeled, unlabeled dataset
    try:
        inputs_x, target_x = iters[0].next()
    except:
        iters[0] = iter(loaders[0])
        inputs_x, target_x = iters[0].next()

    try:
        (inputs_u_w, inputs_u_s), _ = iters[1].next()
    except:
        iters[1] = iter(loaders[1])
        (inputs_u_w, inputs_u_s), _ = iters[1].next()

    inputs = (inputs_x, inputs_u_w, inputs_u_s)
    batch_size = inputs_x.size(0)

    # Transform label to one-hot
    target_x = target_x.cuda(non_blocking=True)
    target_x = torch.zeros(batch_size, 10).cuda().scatter_(1, target_x.view(-1,1).long(), 1)
    target_x = target_x.unsqueeze(2).repeat(1, 1, particles)

    return inputs, target_x, iters


def compute_log_prob(net, inputs, batch_size):
    logits = net(inputs)
    logits_x = logits[:batch_size]
    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
    
    ## Note that weak augmentaitons have detached gradient
    log_px = nn.functional.log_softmax(logits_x, dim=1)
    log_pw = nn.functional.log_softmax(logits_u_w, dim=1)
    log_ps = nn.functional.log_softmax(logits_u_s, dim=1)

    return (log_px, log_pw, log_ps)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

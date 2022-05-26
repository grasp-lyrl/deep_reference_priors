import torch
import torch.nn as nn
import yaml

from utils.data import get_dataloaders
from omegaconf import OmegaConf

from utils.net import create_net
from utils.runner import get_shapes, eval_and_log_metrics, get_minibatch
from utils.runner import compute_h_yx, compute_hy, compute_log_prob
from utils.runner import get_train_objects


def setup(seed):
    """
    Setup pytorch and random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def read_config(fname):
    """
    Read config from yaml file and print it
    """
    with open(fname, "r") as stream:
        cfg = yaml.safe_load(stream)
    print(cfg)
    return OmegaConf.create(cfg)


def train_ref_prior(cfg, dataloaders):
    """
    Train the model on labeled + unlabeled data using Deep Reference priors
    """
    # Create K neural nets which correspond to particles of the refernece prior.
    net = create_net(cfg)

    # Pre-compute an object which is used for efficiently computing h_yx
    reshapes = get_shapes(cfg, nlabs=10)

    lab_loader, unlab_loader, test_loader = dataloaders
    iters = [iter(lab_loader), iter(unlab_loader)]
    optimizer, scheduler, scaler = get_train_objects(cfg)

    for epoch in range(cfg.steps.epochs):

        if (epoch % cfg.steps.test_epoch == 0):
            eval_and_log_metrics(epoch, net, test_loader)

        # Metrics to track for every epoch
        loss_run, ce_run, mask_run = 0, 0, 0
        h_y_run, h_yx_run = 0, 0 

        net.train()
        for idx in range(cfg.steps.updates):
            optimizer.zero_grad(set_to_none=True)

            inputs, target_x, iters = get_minibatch(iters,
                                                    dataloaders[0:2],
                                                    cfg.ref.particles)
            # Fetch image from labeled dataset. Also fetch weak and strong
            # augmentations of the same image from the unlabeled dataset
            inputs_x, inputs_u_w, inputs_u_s = inputs
            bs_x = inputs_x.size(0)
            
            with amp.autocast(enabled=True):

                # Forward prop: compute log-probabilities 
                all_inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
                log_px, log_pw, log_ps = compute_log_prob(net, all_inputs, bs_x)
                log_pw_d = log_pw.detach()  

                # Compute the loss function
                ce_loss = - torch.mean(torch.sum(log_px * target_x, dim=1))
                h_yx, mask = compute_h_yx(log_pw_d, log_ps, cfg)
                h_y = compute_hy(low_pw, log_ps, cfg)
                info_loss =  (h_yx - cfg.ref.α * h_y) 

                # The 1/(1-τ^2) term is justified in the appendix
                loss = ce_loss + (1. / (1 - cfg.ref.τ**2)) * info_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            net.update_ema()
            scheduler.step()

            ce_run += ce_loss.item()
            loss_run += loss.item()
            h_yx_run += h_yx.item()
            h_y_run += h_y.item()
            mask_run += mask.sum().item()

        info = {'epoch': epoch+1,
                'ce_loss': ce_run/idx,
                'loss': loss_run/idx,
                'H_y':  h_y_run/idx,
                'H_yx':  h_yx_run/idx, 
                'masks':  mask_run/idx}
        print(info)

    eval_and_log_metrics("final_epoch", net, test_loader)
    torch.save(net.state_dict(), 'model.pth')


if __name__ == "__main__":
    cfg = read_config("./config/unlabeled.yaml")
    setup(cfg.seed)
    dataloaders = get_dataloaders(cfg)
    train_ref_prior(cfg, dataloaders)

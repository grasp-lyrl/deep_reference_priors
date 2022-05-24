import torch
import torch.cuda.amp as amp
import torch.optim as optim
import torch.nn as nn
import yaml

from utils.data import get_dataloaders
from omegaconf import OmegaConf

from utils.net import create_net
from utils.runner import get_shapes, log_metrics


def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def read_config(fname):
    with open(fname, "r") as stream:
        cfg = yaml.safe_load(stream)
    print(cfg)
    return OmegaConf.create(cfg)


def train_ref_prior(cfg, dataloaders):
    net = create_net(cfg)
    lab_loader, unlab_loader, test_loader = dataloaders
    
    reshapes = get_shapes(cfg, nlabs=10)

    optimizer = optim.SGD(
        net.get_opt_params(wd=cfg.hp.wd / cfg.ref.particles),
        lr=cfg.hp.lr, momentum=0.9, nesterov= True )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, cfg.steps.eval * cfg.steps.epochs)
    scaler = amp.GradScaler(enabled=True)

    for epoch in range(cfg.steps.epochs):

        if (epoch % cfg.steps.test_epoch == 0) or (epoch == cfg.steps.epochs - 1):
            log_metrics(net, test_loader)
            
        iters = (iter(lab_loader), iter(unlab_loader))
        net.train()

        loss_run, ce_run, mask_run = 0.0, 0.0, 0.0
        h_y_run, h_yx_run = 0., 0. 

        for idx in range(cfg.eval_step):
            optimizer.zero_grad(set_to_none=True)

            inputs, target_x, iters = get_minibatch(iters,
                                                    dataloaders[0:1],
                                                    cfg.ref.particles)
            inputs_x, inputs_u_w, inputs_u_s = inputs
            batch_size = inputs_x.size(0)
            
            with amp.autocast(enabled=cfg.fp16):

                all_inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
                log_px, log_pw, log_ps = compute_log_prob(net, all_inputs, batch_size)

                # 1) Cross-entrpoy Loss
                ce_loss = - torch.mean(torch.sum(log_px * targets_x, dim=1))

                # 2) Reference prior objective

                # 2a) Compute H_yx
                # Apply Jensen's inequality here
                p_ave = cfg.ref.τ * log_pw.exp() + (1 - cfg.ref.τ) * log_ps.exp()
                log_p_ave = cfg.ref.τ * log_pw  + (1 - cfg.ref.τ) * log_ps
                entropy = - (p_ave * log_p_ave).sum(1)     
                # Samples contribute to the loss only if their predictions are confident
                max_probs, _ = torch.max(log_pw.exp(), dim=1)
                mask = max_probs.ge(cfg.threshold)    
                h_yx = (entropy * mask).mean()

                # 2b) Compute H_y
                # Same as log_pw (but with gradients)
                ln_p_y = log_softmax(logits_u_w, dim=1)
                ln_p_y = torch.transpose(ln_p_y, 1, 2)
                bs =  batch_size // cfg.ref.order
                ln_p_y = torch.reshape(ln_p_y, (cfg.ref.order, bs, cfg.ref.particles, 10))
                # Sum over all combinations of n particles
                ln_p_yn = [ln_p_y[i].view(shapes[i]) for i in range(cfg.ref.order)]
                ln_p_yn = sum(ln_p_yn).view(bs, cfg.ref.particles, -1)
                # Weights particles by prior and sum
                pi_ln_p = (ln_p_yn.exp() * prior.view(1, -1, 1)).sum(dim=1)
                h_y = - (pi_ln_p * torch.log(pi_ln_p + 1e-12)).sum(1).mean() 
                h_y = h_y / cfg.ref.order

                # 3) Compute the final loss
                loss = ce_loss + (1./( 1 - cfg.ref.τ**2 )) * (h_yx - cfg.ref.α * h_y) 

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            net.update_ema()
            scheduler.step()

            # lr_scheduler is not used
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

    torch.save(net.state_dict(), 'model.pth')


if __name__ == "__main__":
    cfg = read_config("./config/unlabeled.yaml")
    dataloaders = get_dataloaders(cfg)
    train_ref_prior(cfg, dataloaders)

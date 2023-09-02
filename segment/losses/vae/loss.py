import torch
import torch.nn.functional as F

def vqvae_loss(images,vqvae_output,beta = 1.0):
    loss_recons = F.mse_loss(vqvae_output['x_tilde'],images)
    loss_vq = F.mse_loss(vqvae_output['z_q_x'],vqvae_output['z_e_x'].detach())
    loss_commit = F.mse_loss(vqvae_output['z_e_x'],vqvae_output['z_q_x'].detach())

    loss = loss_recons + loss_vq + beta * loss_commit

    return loss
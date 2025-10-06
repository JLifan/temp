import torch
import torch.nn.functional as F


def get_model_device(model):
    try:
        return next(model.parameters()).device.type
    except StopIteration:
        try:
            return next(model.buffers()).device.type
        except StopIteration:
            return None


def add_dp_noise(input, noise_multiplier=1.0):
    noise = torch.normal(mean=0, std=noise_multiplier, size=input.size()).to(input.device)
    return input + noise


def masked_mse_loss(preds, targets, mask):
    mask = mask.unsqueeze(-1)       # (batch_size, seq_length, 1)
    mask = mask.expand_as(preds)    # (batch_size, seq_length, action_dim)

    diff = (preds - targets) ** 2
    masked_diff = diff * mask

    loss = masked_diff.sum() / mask.sum().clamp(min=1.0)
    return loss


def masked_critic_loss(preds, targets, mask, tau=0.5, beta=1):
    u = (targets - preds.mu) / preds.sigma / beta

    nll = torch.sum(
        (torch.abs(tau - (u < 0).float()).squeeze(-1) * preds.nll(targets))[mask > 0]
    ) / mask.sum().clamp(min=1.0)

    return nll * (1 / max(tau, 1 - tau))


def estimate_action_entropy_from_embeddings(action_embeddings, mask=None):
    """
    action_embeddings: [batch_size, action_num, action_dim]
    mask: [batch_size, action_num]
    """
    batch_size, action_num, action_dim = action_embeddings.shape

    # calculate pairwise cosine similarity
    normed = F.normalize(action_embeddings, dim=-1)  # [batch_size, action_num, action_dim]
    similarity = torch.matmul(normed, normed.transpose(1, 2))  # [batch_size, action_num, action_num]

    # Ignore oneself (diagonal) to avoid similarity of 1 affecting entropy
    similarity = similarity - torch.eye(action_num, device=similarity.device).unsqueeze(0)

    # softmax over rows -> simulate a 'similarity distribution'
    sim_dist = F.softmax(similarity, dim=-1)  # [batch_size, action_num, action_num]

    # entropy calculation
    log_sim_dist = torch.log(sim_dist + 1e-8)
    entropy = -torch.sum(sim_dist * log_sim_dist, dim=-1)  # [B, T]

    if mask is not None:
        entropy = entropy * mask
        traj_entropy = entropy.sum(dim=1) / mask.sum(dim=1)
    else:
        traj_entropy = entropy.mean(dim=1)

    return traj_entropy.mean()

def safe_clear_grad_sample(module):
    for submodule in module.modules():
        if hasattr(submodule, "grad_sample"):
            submodule.grad_sample = None
        if hasattr(submodule, "_forward_hooks"):
            submodule._forward_hooks.clear()
        if hasattr(submodule, "_backward_hooks"):
            submodule._backward_hooks.clear()
        if hasattr(submodule, "_forward_pre_hooks"):
            submodule._forward_pre_hooks.clear()

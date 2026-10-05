import torch
import torch.nn as nn
def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
)->torch.Tensor:
    max_logits = torch.max(logits, dim = -1, keepdim = True).values
    shift_logits = logits - max_logits
    exp_logits = torch.exp(shift_logits)
    exp_sum = exp_logits.sum(dim = -1, keepdim = True)
    log_exp_sum = torch.log(exp_sum)
    targets = targets.unsqueeze(-1)
    target_logit = shift_logits.gather(dim = -1, index = targets)
    return (log_exp_sum - target_logit).mean()
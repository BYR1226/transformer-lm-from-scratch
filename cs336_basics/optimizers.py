import torch
import torch.nn as nn
import math
#AdamW优化器
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # 更新参数不应被记录进反向传播计算图
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1

                t = state["t"]
                g = p.grad
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                adjusted_lr = lr*(math.sqrt(1-beta2**t))/(1-beta1**t)
                p.mul_(1 - lr*weight_decay)#直接缩放模型参数
                state["m"] = beta1*state["m"]+(1-beta1)*g
                state["v"] = beta2*state["v"]+(1-beta2)*g**2
                m = state["m"]
                v = state["v"]
                p.add_(- adjusted_lr*m/(v**0.5+eps))

#余弦退火调整学习率
#参数含义：t是当前迭代步数,T_w是Warmup预热步数,T_c是余弦退火截止总步数
def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t*alpha_max/T_w
    elif t >= T_w and t <= T_c:
        return alpha_min + 0.5*(1+math.cos((t-T_w)/(T_c-T_w)*math.pi))*(alpha_max-alpha_min)
    elif t > T_c:
        return alpha_min

#梯度裁剪
@torch.no_grad()
def clip_gradients(parameters, max_norm):
    parameters = list(parameters)
    total_sq = 0
    for p in parameters:
        if p.grad is None:
            continue
        total_sq += (p.grad**2).sum()
    sqrt_sum = total_sq**0.5
    if sqrt_sum > max_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(max_norm/(sqrt_sum + 1e-6))




import torch
import torch.nn as nn
import numpy as np
from modules import TransformerLM
from loss import cross_entropy
from data import get_batch
from checkpointing import save_checkpoint,load_checkpoint
from optimizers import AdamW,get_lr_cosine_schedule,clip_gradients

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #数据配置
    train_data_path: str
    val_data_path: str
    batch_size: int
    context_length: int

    #模型配置
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    theta: float  # RoPE的theta

    #优化器,学习率配置
    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int
    weight_decay: float
    max_grad_norm: float

    #训练控制配置
    max_iters: int
    val_interval: int
    save_interval: int
    ckpt_path: str
    resume_path: str | None = None

    train_data = np.load(train_data_path, mmap_mode="r")
    val_data = np.load(val_data_path, mmap_mode="r")

    model = TransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff,
        theta = theta,
        device = device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr = max_lr,
        betas = (0.9, 0.95),
        eps = 1e-8,
        weight_decay = weight_decay
    )
    #恢复训练
    start_iteration = 0
    if resume_path is not None:
        start_iteration = load_checkpoint(resume_path, model, optimizer)
    model.train()

    #训练主循环
    for iteration in range(start_iteration, max_iters):
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), max_grad_norm)
        current_lr = get_lr_cosine_schedule(iteration, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        optimizer.step()

        if (iteration + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch(val_data, batch_size, context_length, device)
                val_logits = model(val_x)
                val_loss = cross_entropy(val_logits, val_y)
            #打印日志
            print(
                f"iteration={iteration + 1}, "
                f"train_loss={loss.item():.4f}, "
                f"val_loss={val_loss.item():.4f}"
            )
            model.train()

        if (iteration + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, iteration+1, ckpt_path)



if __name__ == "__main__":
    main()

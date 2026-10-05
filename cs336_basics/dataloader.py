import torch
import torch.nn as nn
import numpy as np

def get_batch(data, batch_size, context_length, device):
    high = len(data) - context_length
    low = 0
    size = batch_size
    #start_indices是一个一维 NumPy整数数组，里面保存这个batch中每条训练序列的随机起点
    start_indices = np.random.randint(low=low, high=high, size=size)
    x_array = np.stack(
        [data[index:index+context_length] for index in start_indices]
    )
    y_array = np.stack(
        [data[index+1:index + context_length+1] for index in start_indices]
    )
    x_array = torch.tensor(x_array, dtype = torch.int64, device = device)
    y_array = torch.tensor(y_array, dtype = torch.int64, device = device)
    return x_array, y_array


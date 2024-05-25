import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor, Tensor


def skip_gram(walk: list, max_window_size, optimizer: optim.Optimizer, model):
    running_loss = 0.0
    cnt = 0
    # soft window
    w = random.randint(1, max_window_size)
    for j, v in enumerate(walk):
        window = walk[j - w: j] + walk[j + 1: j + w + 1]
        for u in window:
            # reset gradient về 0
            optimizer.zero_grad()
            # tính hàm mất mát
            loss: Tensor = model(u, v)
            # lan truyền ngược để tính gradient cho các parameter
            loss.backward()
            # tối ưu các parameter với gradient tính đc
            optimizer.step()

            # Ghi nhận lại hàm mất mát
            running_loss += loss.item()
            cnt += 1
    return (running_loss, cnt)

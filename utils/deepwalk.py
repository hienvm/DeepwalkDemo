import random
import torch
import torch.nn as nn
from torch import Tensor, tensor
import torch.optim as optim
import torch.multiprocessing as mp
import tqdm

from utils.similarity import similarity
from utils.walks import random_walk


class DeepwalkModel(nn.Module):
    def __init__(self, V: int, emb_sz: int, leaf_pos: list) -> None:
        super(DeepwalkModel, self).__init__()
        # Hidden layer biểu diễn vector cho đỉnh
        self.embedding_layer = nn.Embedding(
            num_embeddings=V, embedding_dim=emb_sz
        )

        # Biểu diễn T bằng complete binary tree đếm từ 0
        # Luôn biểu diễn được T sao cho số lá tầng cuối = chẵn
        # <=> số nút trong = số nút lá (tức V) - 1
        self.inner_nodes_cnt = V - 1
        # Output layer (các nút trong của T) cho hierarchical softmax
        self.hsoftmax_layer = nn.Parameter(
            torch.rand((self.inner_nodes_cnt, emb_sz))
        )

        # thứ tự lá
        self.leaf_pos = leaf_pos

    def forward(self, u: int, v: int):
        """Đoán Pr(u | v_emb), trả về loss function"""
        loss = tensor(0.0)

        # Lấy ra biểu diễn vector của v
        v_emb = self.embedding_layer(tensor(v))

        # Tính nút lá ứng với u trong cây nhị phân T
        node = self.inner_nodes_cnt + self.leaf_pos[u]

        while node:
            # Kiểm tra nút hiện tại là con trái hay phải
            isLeftChild = node & 1

            # Nhảy lên nút cha
            if isLeftChild:
                node >>= 1
            else:
                node = (node - 1) >> 1

            # Lấy ra biểu diễn vector của nút trong
            node_emb = self.hsoftmax_layer[tensor(node)]

            # cập nhật loss function trên đường đi tới gốc
            if isLeftChild:
                loss -= similarity(node_emb, v_emb).sigmoid().log()
            else:
                loss -= (
                    tensor(1.0)
                    - similarity(node_emb, v_emb).sigmoid()
                ).log()
        return loss


def train_deepwalk(
    model: nn.Module,
    start_lr: float,
    end_lr: float,
    vertices: list,
    adj_lists: list[list],
    loss_records: list,
    walk_len: int,
    walks_per_vertex: int,
    window_sz: int,
    worker_threads: int,
    chunk_sz: int,
):
    lr_step = (start_lr - end_lr) / walks_per_vertex
    for i in tqdm.tqdm(range(walks_per_vertex)):
        epoch_loss = 0.0
        cnt = 0
        random.shuffle(vertices)
        if worker_threads > 1:
            with mp.Pool() as pool:
                # sinh ra corpus và chia thành một batch cho mỗi walk từ một đỉnh nguồn
                batches = [
                    (
                        random_walk(v, adj_lists, walk_len),
                        window_sz,
                        start_lr,
                        model,
                    ) for v in vertices
                ]
                # huấn luyện skipgram
                results = pool.starmap(
                    func=skip_gram_deepwalk,
                    iterable=batches,
                    chunksize=chunk_sz,
                )
            for (running_loss, running_cnt) in results:
                # Ghi nhận lại hàm mất mát
                epoch_loss += running_loss
                cnt += running_cnt
        else:
            for src in vertices:
                # lấy ra đường đi ngẫu nhiên
                walk = random_walk(src, adj_lists, walk_len)
                # huấn luyện skipgram
                (running_loss, running_cnt) = skip_gram_deepwalk(
                    walk=walk,
                    window_sz=window_sz,
                    lr=start_lr,
                    model=model
                )
                # Ghi nhận lại hàm mất mát
                epoch_loss += running_loss
                cnt += running_cnt

        # tính trung bình hàm mất mát cho lần chạy
        loss_records.append(epoch_loss / cnt)

        # Giảm tuyến tính tỉ lệ học
        start_lr -= lr_step


def skip_gram_deepwalk(walk: list, window_sz: int, lr: float, model: nn.Module):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    running_loss = 0.0
    cnt = 0
    # soft window
    w = random.randint(1, window_sz)
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

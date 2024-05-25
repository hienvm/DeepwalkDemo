import random
import torch
import torch.nn as nn
from torch import tensor

from utils.similarity import similarity
from utils.skipgram import skip_gram
from utils.walks import random_walk


class DeepwalkModel(nn.Module):
    def __init__(self, V, emb_sz, leaf_pos) -> None:
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

    def forward(self, u, v):
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


def train_deepwalk(model, optimizer, vertices, adj_lists, loss_records, scheduler, walk_len, walks_per_vertex, window_sz):
    for i in range(walks_per_vertex):
        epoch_loss = 0.0
        cnt = 0
        random.shuffle(vertices)
        for src in vertices:
            # lấy ra đường đi ngẫu nhiên
            walk = random_walk(src, adj_lists, walk_len)
            # huấn luyện skipgram
            (running_loss, running_cnt) = skip_gram(
                walk=walk, max_window_size=window_sz, optimizer=optimizer, model=model)
            # Ghi nhận lại hàm mất mát
            epoch_loss += running_loss
            cnt += running_cnt

        # tính trung bình hàm mất mát cho lần chạy
        loss_records.append(epoch_loss / cnt)

        # Giảm tuyến tính tỉ lệ học
        scheduler.step()

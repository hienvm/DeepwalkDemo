import random
import torch
import torch.nn as nn
from torch import Tensor, tensor
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm.notebook import tqdm
import numpy as np
from scipy.stats.sampling import DiscreteAliasUrn
from sklearn.preprocessing import normalize
from IPython.display import display

from utils.similarity import similarity
from utils.walks import biased_random_walk


class Node2VecModel(nn.Module):
    def __init__(self, V, emb_sz, k) -> None:
        super(Node2VecModel, self).__init__()
        # Hidden layer biểu diễn vector cho đỉnh
        self.embedding_layer = nn.Embedding(
            num_embeddings=V, embedding_dim=emb_sz)

        self.k = k

    def forward(self, u, v, excludes, neg_rng):
        '''Đoán xác xuất log(Pr(u | v_emb)), trả về loss'''
        # Lấy ra biểu diễn vector
        u_emb: Tensor = self.embedding_layer(tensor(u))
        v_emb: Tensor = self.embedding_layer(tensor(v))

        # chọn negative sample
        sample = get_neg_sample(excludes, self.k, neg_rng)

        # positive
        p = similarity(v_emb, u_emb).sigmoid().log().mul(-1)

        # negative
        for noise in sample:
            noise_emb: Tensor = self.embedding_layer(tensor(noise))
            p -= similarity(v_emb, noise_emb).mul(-1).sigmoid().log()

        return p


def train_node2vec(
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
    noise_distr: np.ndarray,
    transition_rng,
):
    # RNG dùng để lấy ra negative sample
    neg_rng = DiscreteAliasUrn(
        noise_distr, random_state=np.random.default_rng())
    # Tần suất xuất hiện của mỗi đỉnh
    freq = np.zeros((len(adj_lists),))
    total_cnt = 0

    lr_step = (start_lr - end_lr) / walks_per_vertex

    for i in tqdm(range(walks_per_vertex)):
        epoch_loss = 0.0
        cnt = 0
        random.shuffle(vertices)
        if worker_threads > 1:
            with mp.Pool() as pool:
                # sinh ra corpus và chia thành một batch cho mỗi walk từ một đỉnh nguồn
                batches = [
                    (
                        biased_random_walk(
                            v, adj_lists, walk_len, transition_rng),
                        window_sz,
                        start_lr,
                        model,
                        freq,
                        neg_rng,
                    ) for v in vertices
                ]
                # huấn luyện skipgram
                results = pool.starmap(
                    func=skipgram_node2vec,
                    iterable=batches,
                    chunksize=chunk_sz,
                )
            for (running_loss, running_cnt) in results:
                # Ghi nhận lại hàm mất mát
                epoch_loss += running_loss
                cnt += running_cnt
        else:
            for v in vertices:
                # lấy ra đường đi ngẫu nhiên
                walk = biased_random_walk(
                    v, adj_lists, walk_len, transition_rng)
                # huấn luyện skipgram
                (running_loss, running_cnt) = skipgram_node2vec(
                    walk=walk,
                    window_sz=window_sz,
                    lr=start_lr,
                    model=model,
                    freq=freq,
                    neg_rng=neg_rng,
                )
                # Ghi nhận lại hàm mất mát
                epoch_loss += running_loss
                cnt += running_cnt

        # tính trung bình hàm mất mát cho lần chạy
        loss_records.append(epoch_loss / cnt)

        # Giảm tuyến tính tỉ lệ học
        start_lr -= lr_step

        # cập nhật lại noise distribution theo unigram distribution
        total_cnt += cnt
        unigram_distr = (freq / total_cnt) ** 0.75
        noise_distr = normalize([unigram_distr])[0]
        # cập nhật rng
        neg_rng = DiscreteAliasUrn(
            noise_distr, random_state=np.random.default_rng())
    display("Final noise distribution:")
    display(noise_distr)


def get_neg_sample(excludes, k, neg_rng):
    # chọn ra k đỉnh ngọai trừ exclude theo phân bố của neg_rng
    excludes = set(excludes)
    sample = set()
    k_tmp = k
    while k_tmp > 0:
        sample.update(neg_rng.rvs(size=k))
        sample.difference_update(excludes)
        k_tmp = k - len(sample)
    return sample


def skipgram_node2vec(walk: list, window_sz: int, lr: float, model: Node2VecModel, freq: np.ndarray, neg_rng):
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
            loss: Tensor = model(
                u, v, excludes=window+[v], neg_rng=neg_rng)
            # lan truyền ngược để tính gradient cho các parameter
            loss.backward()
            # tối ưu các parameter với gradient tính đc
            optimizer.step()

            # Ghi nhận lại hàm mất mát
            running_loss += loss.item()
            cnt += 1

            # Cập nhật tần suất
            freq[u] += 1
    return (running_loss, cnt)

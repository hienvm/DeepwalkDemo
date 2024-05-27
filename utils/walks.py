import random


def random_walk(v, adj_lists, walk_len):
    '''deepwalk'''
    # walk bắt đầu từ v
    walk = [v]
    for i in range(walk_len - 1):
        # chọn đỉnh kề ngẫu nhiên
        next_node = random.choice(adj_lists[v])
        # thêm đỉnh kề vào walk
        walk.append(next_node)
        # nhảy tới đỉnh kề
        v = next_node
    return walk


def biased_random_walk(v, adj_lists, walk_len, transition_rng):
    # walk bắt đầu từ v
    walk = [v]
    t = v
    for i in range(walk_len - 1):
        # chọn đỉnh kề ngẫu nhiên
        x = adj_lists[v][transition_rng[v][t].rvs()]
        # thêm đỉnh kề vào walk
        walk.append(x)
        # nhảy tới đỉnh kề
        v = x
        t = v

    return walk

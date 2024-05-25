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

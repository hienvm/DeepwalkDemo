from torch import tensor, Tensor


def similarity(u: Tensor, v: Tensor) -> Tensor:
    # có thể dùng khoảng cách euclid cho 1D
    # if u.dim() == 0:
    #     return u.subtract(v).abs()
    return u.dot(v)

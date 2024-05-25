from torch import tensor, Tensor


def similarity(u: Tensor, v: Tensor) -> Tensor:
    # Dùng khoảng cách euclid cho 1D
    if u.dim() == 0:
        return u.subtract(v).abs()
    return u.dot(v)


print(similarity(tensor([1, 2]), tensor([2, 3])))

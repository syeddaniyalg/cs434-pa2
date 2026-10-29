import torch


GROUP_NAMES = ("R0", "R1", "R2", "R3", "MR0", "MR1", "MR2", "MR3")


def mirror(x):
    return torch.flip(x, dims=(-1,))


def rotate(x, turns=1):
    return torch.rot90(x, k=int(turns) % 4, dims=(-2, -1))


def apply_group(x, group):
    group = int(group)
    turns = group % 4
    transformed = rotate(x, turns)
    if group >= 4:
        transformed = mirror(transformed)
    return transformed


def invert_group(x, group):
    group = int(group)
    if group >= 4:
        return apply_group(x, group)
    return rotate(x, -group)


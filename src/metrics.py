import torch


def masked_training_loss(predictions, targets, mask):
    horizon_errors = (predictions - targets).square().mean(dim=(2, 3, 4))
    weights = mask.to(horizon_errors.dtype)
    per_video = (horizon_errors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return per_video.mean()


def per_video_horizon_mse(predictions, targets):
    return (predictions - targets).square().mean(dim=(2, 3, 4))


def masked_per_video_error(predictions, targets, mask, horizons=None):
    errors = per_video_horizon_mse(predictions, targets)
    if horizons is not None:
        errors = errors[:, :horizons]
        mask = mask[:, :horizons]
    weights = mask.to(errors.dtype)
    return (errors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def per_video_consistency(first, second, horizons=5):
    return (first[:, :horizons] - second[:, :horizons]).square().mean(dim=(1, 2, 3, 4))


def aggregate_horizon_mse(values):
    return values.mean(dim=0)


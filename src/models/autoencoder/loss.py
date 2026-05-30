import torch.nn as nn


class VariationalMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(VariationalMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.mse_loss(
            x_hat, x, reduction=self.reduction
        )

        kl_divergence = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

        if self.reduction == "none":
            return reproduction_loss + kl_divergence.mean()
        elif self.reduction == "sum":
            return reproduction_loss + kl_divergence.sum()
        else:
            return reproduction_loss + kl_divergence.mean()


class WarmupBetaVariationalMSELoss(nn.Module):
    """β-annealed VAE loss for posterior-collapse mitigation.

    During the first `warmup_epochs` of training, β linearly anneals from 0
    to `target_beta`. Lets the encoder establish meaningful latent
    structure before KL pressure kicks in. After warmup, β stays at
    `target_beta` (1.0 = standard VAE).

    Usage: instantiate with target_beta=1.0 for "standard VAE with KL
    warmup". Subclass with target_beta < 1.0 for "β-VAE with warmup".

    The training loop must update `current_epoch` between epochs (see
    BaseTrainingRegime — adapter accommodates losses with that attribute).
    """

    def __init__(self, reduction: str = "mean",
                 target_beta: float = 1.0,
                 warmup_epochs: int = 20) -> None:
        super().__init__()
        self.reduction = reduction
        self.target_beta = target_beta
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def forward(self, x, x_hat, mean, log_var):
        beta = min(self.target_beta,
                   self.target_beta * self.current_epoch / max(1, self.warmup_epochs))
        reproduction_loss = nn.functional.mse_loss(
            x_hat, x, reduction=self.reduction
        )
        kl_divergence = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        if self.reduction == "none":
            return reproduction_loss + beta * kl_divergence.mean()
        elif self.reduction == "sum":
            return reproduction_loss + beta * kl_divergence.sum()
        else:
            return reproduction_loss + beta * kl_divergence.mean()

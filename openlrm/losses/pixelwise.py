import torch
import torch.nn as nn

__all__ = ['PixelLoss']


class PixelLoss(nn.Module):
    """
    Pixel-wise loss between two images.
    """

    def __init__(self, option: str = 'mse'):
        super().__init__()
        self.loss_fn = self._build_from_option(option)

    @staticmethod
    def _build_from_option(option: str, reduction: str = 'none'):
        if option == 'mse':
            return nn.MSELoss(reduction=reduction)
        elif option == 'l1':
            return nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError(f'Unknown pixel loss option: {option}')

    @torch.compile
    def forward(self, x, y):
        """
        Assume images are channel first.
        
        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
        
        Returns:
            Mean-reduced pixel loss across batch.
        """
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        y = y.reshape(N*M, C, H, W)
        image_loss = self.loss_fn(x, y).mean(dim=[1, 2, 3])
        batch_loss = image_loss.reshape(N, M).mean(dim=1)
        all_loss = batch_loss.mean()
        return all_loss

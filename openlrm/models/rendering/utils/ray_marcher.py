import torch
import torch.nn as nn


class MipRayMarcher2(nn.Module):
    def __init__(self, activation_factory):
        super().__init__()
        self.activation_factory = activation_factory

    def run_forward(self, colors, densities, depths, rendering_options, bg_colors=None):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        densities_mid = self.activation_factory(rendering_options)(densities_mid)

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total


        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total
        else:
            assert bg_colors is not None, "Must provide bg_colors if white_back is False"
            composite_rgb = composite_rgb + bg_colors.unsqueeze(-1) * (1 - weight_total)




        return composite_rgb, composite_depth, weights


    def forward(self, colors, densities, depths, rendering_options, bg_colors=None):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options, bg_colors=bg_colors)

        return composite_rgb, composite_depth, weights

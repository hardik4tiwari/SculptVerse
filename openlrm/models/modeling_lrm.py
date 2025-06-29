import torch
import torch.nn as nn
from accelerate.logging import get_logger

from .embedder import CameraEmbedder
from .transformer import TransformerDecoder
from .rendering.synthesizer import TriplaneSynthesizer


logger = get_logger(__name__)


class ModelLRM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, camera_embed_dim: int, rendering_samples_per_ray: int,
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768):
        super().__init__()
        

        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim


        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12+4, embed_dim=camera_embed_dim,
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type='cond_mod',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=camera_embed_dim,
        )
        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=2, stride=2, padding=0)
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim, samples_per_ray=rendering_samples_per_ray,
        )

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in ['dino', 'dinov2'], "Unsupported encoder type"
        if encoder_type == 'dino':
            from .encoders.dino_wrapper import DinoWrapper
            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == 'dinov2':
            from .encoders.dinov2_wrapper import Dinov2Wrapper
            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper

    def forward_transformer(self, image_feats, camera_embeddings):
        assert image_feats.shape[0] == camera_embeddings.shape[0], \
            "Batch size mismatch for image_feats and camera_embeddings!"
        N = image_feats.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)
        x = x.contiguous().view(3*N, -1, H, W)
        x = self.upsampler(x)
        x = x.view(3, N, *x.shape[-3:])
        x = torch.einsum('indhw->nidhw', x)
        x = x.contiguous()
        return x

    @torch.compile
    def forward_planes(self, image, camera):


        N = image.shape[0]


        image_feats = self.encoder(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"


        camera_embeddings = self.camera_embedder(camera)
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"


        tokens = self.forward_transformer(image_feats, camera_embeddings)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes

    def forward(self, image, source_camera, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size: int):







        assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
        assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
        assert image.shape[0] == render_anchors.shape[0], "Batch size mismatch for image and render_anchors"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        N, M = render_cameras.shape[:2]

        planes = self.forward_planes(image, source_camera)


        render_results = self.synthesizer(planes, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size)
        assert render_results['images_rgb'].shape[0] == N, "Batch size mismatch for render_results"
        assert render_results['images_rgb'].shape[1] == M, "Number of rendered views should be consistent with render_cameras"

        return {
            'planes': planes,
            **render_results,
        }

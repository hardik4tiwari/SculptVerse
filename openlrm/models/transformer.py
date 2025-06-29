from functools import partial
import torch
import torch.nn as nn
from accelerate.logging import get_logger


logger = get_logger(__name__)


class TransformerDecoder(nn.Module):

    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    def __init__(self, block_type: str,
                 num_layers: int, num_heads: int,
                 inner_dim: int, cond_dim: int = None, mod_dim: int = None,
                 eps: float = 1e-6):
        super().__init__()
        self.block_type = block_type
        self.layers = nn.ModuleList([
            self._block_fn(inner_dim, cond_dim, mod_dim)(
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in ['basic', 'cond', 'mod', 'cond_mod'], \
            f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == 'basic':
            assert cond_dim is None and mod_dim is None, \
                f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock
            logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == 'cond':
            assert cond_dim is not None, f"Condition dimension must be specified for ConditionBlock"
            assert mod_dim is None, f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock
            logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == 'mod':
            logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(f"modulation without condition is not implemented")
        elif self.block_type == 'cond_mod':
            assert cond_dim is not None and mod_dim is not None, \
                f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock
            logger.debug(f"Using ConditionModulationBlock")
            return partial(ConditionModulationBlock, inner_dim=inner_dim, cond_dim=cond_dim, mod_dim=mod_dim)
        else:
            raise ValueError(f"Unsupported block type during runtime: {self.block_type}")

    def assert_runtime_integrity(self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == 'basic':
            assert cond is None and mod is None, \
                f"Condition and modulation are not supported for BasicBlock"
        elif self.block_type == 'cond':
            assert cond is not None and mod is None, \
                f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == 'mod':
            raise NotImplementedError(f"modulation without condition is not implemented")
        else:
            assert cond is not None and mod is not None, \
                f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor):
        if self.block_type == 'basic':
            return layer(x)
        elif self.block_type == 'cond':
            return layer(x, cond)
        elif self.block_type == 'mod':
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, mod: torch.Tensor = None):



        self.assert_runtime_integrity(x, cond, mod)
        for layer in self.layers:
            x = self.forward_layer(layer, x, cond, mod)
        x = self.norm(x)
        return x

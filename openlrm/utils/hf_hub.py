














import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def wrap_model_hub(model_cls: nn.Module):
    class HfModel(model_cls, PyTorchModelHubMixin):
        def __init__(self, config: dict):
            super().__init__(**config)
            self.config = config
    return HfModel

import math
from functools import partial
import torch

__all__ = ['MixerDataset']


class MixerDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 split: str,
                 subsets: list[dict],
                 **dataset_kwargs,
                 ):
        self.subsets = [
            self._dataset_fn(subset, split)(**dataset_kwargs)
            for subset in subsets
        ]
        self.virtual_lens = [
            math.ceil(subset_config['sample_rate'] * len(subset_obj))
            for subset_config, subset_obj in zip(subsets, self.subsets)
        ]

    @staticmethod
    def _dataset_fn(subset_config: dict, split: str):
        name = subset_config['name']

        dataset_cls = None
        if name == "objaverse":
            from .objaverse import ObjaverseDataset
            dataset_cls = ObjaverseDataset



        else:
            raise NotImplementedError(f"Dataset {name} not implemented")

        return partial(
            dataset_cls,
            root_dirs=subset_config['root_dirs'],
            meta_path=subset_config['meta_path'][split],
        )

    def __len__(self):
        return sum(self.virtual_lens)

    def __getitem__(self, idx):
        subset_idx = 0
        virtual_idx = idx
        while virtual_idx >= self.virtual_lens[subset_idx]:
            virtual_idx -= self.virtual_lens[subset_idx]
            subset_idx += 1
        real_idx = virtual_idx % len(self.subsets[subset_idx])
        return self.subsets[subset_idx][real_idx]

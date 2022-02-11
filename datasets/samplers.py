from collections import defaultdict
from copy import copy, deepcopy
from typing import List, Optional

import torch
from torch.distributions import Categorical
from torch.utils.data import Sampler, RandomSampler, Subset, Dataset
from tqdm import tqdm


class HardSampler(Sampler[List[int]]):
    def __init__(self, data_source: Dataset, batch_size: int, valid_indices=None, replacement: bool = False, num_hard_samples=2,
                 num_samples: Optional[int] = None, generator=None, drop_last=False) -> None:
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.valid_indices = valid_indices
        self.num_hard_samples = num_hard_samples
        self.standard_sampler = RandomSampler(data_source=Subset(self.data_source, valid_indices), replacement=replacement,
                                              num_samples=num_samples,
                                              generator=generator)
        self.current_hard_indices = range(len(self.data_source))
        self.next_hard_indices = []
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.standard_sampler:
            if len(batch)<= self.num_hard_samples and len(self.current_hard_indices) >= 0:
                batch.append(self.current_hard_indices[torch.randint(low=0, high=len(self.current_hard_indices),size=(1,))])
            else:
                batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def add_hard_indices(self, indices):
        self.next_hard_indices.extend(indices)

    def set_hard_indices(self):
        self.current_hard_indices = deepcopy(self.next_hard_indices)
        self.next_hard_indices = []

    def __len__(self):
        # Can only be called if self.standard_sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return len(self.standard_sampler)  # type: ignore

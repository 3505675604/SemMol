from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.datasets.collator import PretrainingDataCollator
from src.datasets.loader import create_dataloader, set_dataloader_epoch


class _EpochAwareCollator:
    def __init__(self) -> None:
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, rows):
        return torch.stack([row[0] for row in rows])


def test_set_dataloader_epoch_updates_epoch_aware_collator() -> None:
    collator = _EpochAwareCollator()
    loader = create_dataloader(
        TensorDataset(torch.arange(4)),
        batch_size=2,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    set_dataloader_epoch(loader, 5)

    assert collator.epoch == 5


class _SampleDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self._sample = {
            "sample_id": "pcqm:1",
            "source_index": 1,
            "smiles": "CC",
            "input_ids": torch.tensor([2, 8, 9, 10, 11, 3]),
            "token_spans": torch.tensor(
                [[-1, -1], [0, 1], [0, 1], [1, 2], [1, 2], [-1, -1]]
            ),
            "atomic_numbers": torch.tensor([6, 6]),
            "coords": torch.zeros((1, 2, 3), dtype=torch.float32),
            "conformer_mask": torch.tensor([True]),
            "energies": torch.tensor([0.0]),
            "energy_mask": torch.tensor([True]),
            "heavy_atom_indices": torch.tensor([0, 1]),
            "conformer_sources": ("unit",),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self._sample


def test_epoch_reaches_persistent_collate_worker() -> None:
    collator = PretrainingDataCollator(
        pad_token_id=0,
        mask_token_id=4,
        vocab_size=32,
        special_token_ids=(0, 1, 2, 3, 4),
        smiles_mask_ratio=0.5,
        node_mask_ratio=0.5,
        edge_mask_ratio=0.5,
        geo_noise_std=0.2,
        seed=3407,
    )
    loader = create_dataloader(
        _SampleDataset(),
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
    )

    set_dataloader_epoch(loader, 0)
    epoch_zero = next(iter(loader))
    set_dataloader_epoch(loader, 1)
    epoch_one = next(iter(loader))

    assert not torch.equal(epoch_zero["coord_noise"], epoch_one["coord_noise"])

"""Read the author's 7x7 MATLAB patches without materializing the test set."""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

CHANNELS = {"augsburg": {"hsi": 180, "sar": 4, "dsm": 1},
            "houston2013": {"hsi": 144, "ms": 8, "lidar": 1}}
CLASSES = {"augsburg": 7, "houston2013": 15}


class MatMatrix:
    def __init__(self, path: Path, samples: int | None = None):
        self.path = path
        self.handle = None
        if h5py.is_hdf5(path):
            self.handle = h5py.File(path, "r")
            keys = [k for k in self.handle if not k.startswith("#")]
            if len(keys) != 1:
                raise ValueError(f"Ambiguous MAT variables: {path}: {keys}")
            self.array = self.handle[keys[0]]
            # MATLAB v7.3 stores dimensions in reverse order in HDF5.
            self.transpose = True
        else:
            arrays = {k: v for k, v in scipy.io.loadmat(path).items() if not k.startswith("__")}
            if len(arrays) != 1:
                raise ValueError(f"Ambiguous MAT variables: {path}: {list(arrays)}")
            self.array = next(iter(arrays.values()))
            self.transpose = False
        self.shape = tuple(reversed(self.array.shape)) if self.transpose else self.array.shape
        if samples is not None and self.shape[0] != samples:
            raise ValueError(f"Sample axis mismatch: {path}: {self.shape}, expected {samples}")

    def row(self, index):
        return np.asarray(self.array[:, index] if self.transpose else self.array[index, :])

    def labels(self):
        return np.asarray(self.array).reshape(-1).astype(np.int64) - 1

    def close(self):
        if self.handle is not None:
            self.handle.close()


class AuthorPatches(Dataset):
    def __init__(self, root: Path, dataset: str, modalities: list[str], split="train"):
        self.dataset = dataset
        self.modalities = modalities
        folder = root / dataset
        self.label_matrix = MatMatrix(folder / modalities[0] / f"{modalities[0]}_Y_{split}.mat")
        self.labels = self.label_matrix.labels()
        if len(self.labels) == 0 or self.labels.min() < 0 or self.labels.max() >= CLASSES[dataset]:
            raise ValueError("Labels must use MATLAB's 1-based class indices")
        self.arrays = {}
        for modal in modalities:
            with_label = MatMatrix(folder / modal / f"{modal}_Y_{split}.mat")
            try:
                if not np.array_equal(self.labels, with_label.labels()):
                    raise ValueError(f"Modalities are not sample-aligned: {modal}")
            finally:
                with_label.close()
            matrix = MatMatrix(folder / modal / f"{modal}_X_{split}.mat", len(self.labels))
            expected = 7 * 7 * CHANNELS[dataset][modal]
            if len(matrix.shape) != 2 or matrix.shape[1] != expected:
                raise ValueError(f"Expected Nx{expected} patches, got {matrix.shape}")
            self.arrays[modal] = matrix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {}
        for modal, matrix in self.arrays.items():
            # Same MATLAB/Fortran reshape as the original datasets/*.py.
            patch = matrix.row(index).reshape(7, 7, CHANNELS[self.dataset][modal], order="F")
            patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
            if not np.isfinite(patch).all():
                raise ValueError(f"Non-finite input in {modal}, sample {index}")
            sample[modal] = torch.from_numpy(patch)
        sample["label"] = torch.tensor(self.labels[index], dtype=torch.long)
        return sample

    def close(self):
        self.label_matrix.close()
        for matrix in self.arrays.values():
            matrix.close()


class AugsburgScenePatches(Dataset):
    """Equivalent Augsburg preprocessing, with patches cropped on demand.

    Per-band scene min/max and MATLAB column-major label order were checked
    against EVERY author training patch for HS, SAR and DSM (761 per modality).
    This uses the author's scene-wide normalization convention; it is not a new
    train-only normalization protocol. The 4.54 GB expanded test MAT is unnecessary.
    """
    def __init__(self, project_root: Path, modalities: list[str], split="test"):
        self.modalities = modalities
        self.dataset = "augsburg"
        folder = project_root / "data/raw/Datasets/HS-SAR-DSM Augsburg"
        key = "TrainImage" if split == "train" else "TestImage"
        labels = scipy.io.loadmat(folder / f"{key}.mat")[key]
        columns, rows = np.nonzero(labels.T)
        self.coordinates = np.column_stack([rows, columns])
        self.labels = labels[rows, columns].astype(np.int64) - 1
        files = {"hsi": ("data_HS_LR.mat", "data_HS_LR"),
                 "sar": ("data_SAR_HR.mat", "data_SAR_HR"), "dsm": ("data_DSM.mat", "data_DSM")}
        self.scenes = {}
        for modal in modalities:
            filename, variable = files[modal]
            # One unexpanded scene at a time, about 114 MiB float32 for all HSI bands.
            scene = scipy.io.loadmat(folder / filename)[variable].astype(np.float64)
            if scene.ndim == 2:
                scene = scene[..., None]
            low, high = scene.min(axis=(0, 1)), scene.max(axis=(0, 1))
            scene = ((scene - low) / np.where(high > low, high - low, 1)).astype(np.float32)
            if not np.isfinite(scene).all():
                raise ValueError(f"Non-finite scene: {modal}")
            self.scenes[modal] = scene

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row, col = self.coordinates[index]
        sample = {}
        for modal, scene in self.scenes.items():
            rows = np.clip(np.arange(row - 3, row + 4), 0, scene.shape[0] - 1)
            cols = np.clip(np.arange(col - 3, col + 4), 0, scene.shape[1] - 1)
            patch = scene[rows[:, None], cols[None, :], :]
            sample[modal] = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        sample["label"] = torch.tensor(self.labels[index], dtype=torch.long)
        return sample

    def close(self):
        self.scenes.clear()


def open_official_test(project_root: Path, dataset: str, modalities: list[str]):
    if dataset == "augsburg":
        return AugsburgScenePatches(project_root, modalities, split="test")
    return AuthorPatches(project_root / "data/processed", dataset, modalities, split="test")

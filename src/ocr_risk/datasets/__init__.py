"""Source-document adapters and the dataset/license registry.

Layer 2. Importing this package registers every built-in adapter, so
``build_dataset(id)`` works without the caller knowing which module defines it.
"""

from __future__ import annotations

from ocr_risk.datasets import cord as _cord  # noqa: F401  (registration side effect)
from ocr_risk.datasets import funsd as _funsd  # noqa: F401
from ocr_risk.datasets import ocrd_sbb as _ocrd  # noqa: F401
from ocr_risk.datasets import synthetic as _synthetic  # noqa: F401
from ocr_risk.datasets.base import DatasetAdapter, LicenseSpec, PreflightReport
from ocr_risk.datasets.cord import CordDataset
from ocr_risk.datasets.download import ChecksumMismatchError, DownloadSpec, download
from ocr_risk.datasets.funsd import FunsdDataset
from ocr_risk.datasets.ocrd_sbb import OcrdSbbDataset
from ocr_risk.datasets.registry import available_datasets, build_dataset, register_dataset
from ocr_risk.datasets.synthetic import SyntheticCorpus, SyntheticDataset, shingle_hash

__all__ = [
    "ChecksumMismatchError",
    "CordDataset",
    "DatasetAdapter",
    "DownloadSpec",
    "FunsdDataset",
    "LicenseSpec",
    "OcrdSbbDataset",
    "PreflightReport",
    "SyntheticCorpus",
    "SyntheticDataset",
    "available_datasets",
    "build_dataset",
    "download",
    "register_dataset",
    "shingle_hash",
]

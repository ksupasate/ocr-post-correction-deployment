from __future__ import annotations

from copy import deepcopy

from PIL import Image, ImageDraw

from ocr_risk.datasets.identity import (
    average_hash,
    difference_hash,
    near_document_identity,
    normalized_pixel_sha256,
    text_simhash,
)
from ocr_risk.datasets.qualification import derive_c1_status


def _passing_inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    acquisition = {
        "verified": True,
        "failures": [],
        "license": {"registry_matches_source": True},
    }
    prior = {
        "n_documents": 401,
        "expected_prior_universe": 401,
        "duplicate_ids": 0,
        "missing_from_expected": [],
    }
    pool = {
        "n_documents": 900,
        "documents_by_split": {"train": 800, "validation": 100},
        "expected_documents_by_split": {"train": 800, "validation": 100},
        "duplicate_ids": 0,
        "identity_signals_complete": True,
        "extracted_files": 1800,
        "pre_role_access_ids_all_present": True,
    }
    duplicates = {"complete": True}
    scan = {"clean": True}
    overlap = {"id_overlap": [], "exact_same_document_pool_ids": []}
    return acquisition, prior, pool, duplicates, scan, overlap


def test_c1_pass_exists_only_after_every_input_passes() -> None:
    status, reasons = derive_c1_status(*_passing_inputs())
    assert status == "C1_PASS"
    assert reasons == []


def test_checksum_mismatch_fails_c1() -> None:
    inputs = list(deepcopy(_passing_inputs()))
    inputs[0]["verified"] = False
    inputs[0]["failures"] = [{"kind": "checksum_mismatch", "file": "shard.parquet"}]
    status, reasons = derive_c1_status(*inputs)
    assert status == "C1_FAIL"
    assert reasons == ["checksum_mismatch"]


def test_dataset_overlap_fails_c1() -> None:
    inputs = list(deepcopy(_passing_inputs()))
    inputs[5]["exact_same_document_pool_ids"] = ["synthetic-overlap-document"]
    status, reasons = derive_c1_status(*inputs)
    assert status == "C1_FAIL"
    assert reasons == ["historical_document_overlap"]


def test_unestablished_licence_is_inconclusive() -> None:
    inputs = list(deepcopy(_passing_inputs()))
    inputs[0]["verified"] = False
    inputs[0]["license"]["registry_matches_source"] = False
    status, reasons = derive_c1_status(*inputs)
    assert status == "C1_INCONCLUSIVE"
    assert reasons == ["license_not_established"]


def test_near_historical_match_does_not_silently_fail_or_enter_the_pool() -> None:
    inputs = list(deepcopy(_passing_inputs()))
    inputs[5]["near_same_document_pool_ids"] = ["synthetic-near-document"]
    status, _ = derive_c1_status(*inputs)
    assert status == "C1_PASS"


def _receipt_like_image() -> Image.Image:
    image = Image.new("L", (80, 120), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 10, 70, 15), fill=0)
    draw.rectangle((8, 24, 55, 29), fill=30)
    draw.rectangle((8, 40, 68, 45), fill=0)
    draw.rectangle((45, 90, 70, 96), fill=0)
    return image


def _identity_record(image: Image.Image, text: str) -> dict[str, object]:
    return {
        "width": image.width,
        "height": image.height,
        "ahash": f"{average_hash(image):016x}",
        "dhash": f"{difference_hash(image):064x}",
        "gt_text_simhash": f"{text_simhash(text):016x}",
    }


def test_normalized_pixel_hash_ignores_image_mode_encoding() -> None:
    grayscale = _receipt_like_image()
    rgb = grayscale.convert("RGB")
    assert normalized_pixel_sha256(grayscale) == normalized_pixel_sha256(rgb)


def test_near_identity_requires_visual_and_text_agreement() -> None:
    original = _receipt_like_image()
    brightened = original.point(lambda value: min(255, value + 8))
    left = _identity_record(original, "TOTAL 125000 CASH 150000")
    right = _identity_record(brightened, "TOTAL 125000 CASH 150000")
    unrelated_text = _identity_record(brightened, "INVOICE ALPHA BETA GAMMA")
    assert near_document_identity(left, right)
    assert not near_document_identity(left, unrelated_text)

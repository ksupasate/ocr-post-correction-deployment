"""Image crops as reproducible recipes.

A crop is fully determined by the source image, the box, and the padding/resize/colour
policy. Storing that recipe instead of the pixels buys three things at once (ADR-005):

- **licensing** — no derivative of a non-redistributable corpus enters the artifact tree;
- **size** — the artifact stays small enough to keep on a nearly full disk;
- **reproducibility** — the same recipe reconstructs bit-identical pixels years later, so
  "what did the verifier actually see?" has an exact answer.

Materialized pixels live in a disposable cache keyed by the recipe hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from ocr_risk.io.hashing import canonical_hash
from ocr_risk.io.paths import cache_root
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.evidence import CropRecipe

__all__ = ["CropPolicy", "build_recipe", "crop_path", "materialize"]


@dataclass(frozen=True, slots=True)
class CropPolicy:
    """How a span's box becomes an image crop. Part of the experiment configuration."""

    padding_ratio: float = 0.25
    """Padding as a fraction of the box's larger side. Context around a span is often what
    disambiguates an edit, so how much of it the verifier sees is a parameter, not a
    constant."""
    padding_min_px: int = 4
    target_height: int | None = 48
    grayscale: bool = True


def build_recipe(image_sha256: str, bbox: BBox, policy: CropPolicy) -> CropRecipe:
    """Describe a crop, and hash the description."""
    payload = {
        "image_sha256": image_sha256,
        "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
        "padding_ratio": policy.padding_ratio,
        "padding_min_px": policy.padding_min_px,
        "target_height": policy.target_height,
        "grayscale": policy.grayscale,
    }
    return CropRecipe(
        image_sha256=image_sha256,
        bbox=bbox,
        padding_ratio=policy.padding_ratio,
        padding_min_px=policy.padding_min_px,
        target_height=policy.target_height,
        grayscale=policy.grayscale,
        recipe_sha256=canonical_hash(payload),
    )


def crop_path(recipe: CropRecipe) -> Path:
    """Cache location for a materialized crop. Safe to delete at any time."""
    digest = recipe.recipe_sha256
    return cache_root() / "crops" / digest[:2] / f"{digest}.png"


def _padded_box(recipe: CropRecipe, width: int, height: int) -> tuple[int, int, int, int]:
    box = recipe.bbox
    pad = max(recipe.padding_min_px, recipe.padding_ratio * max(box.width, box.height))
    # Clamp to the page: a box near the margin must not silently shift the crop inward,
    # which would change what the verifier sees depending on where the span sits.
    return (
        int(max(0, round(box.x0 - pad))),
        int(max(0, round(box.y0 - pad))),
        int(min(width, round(box.x1 + pad))),
        int(min(height, round(box.y1 + pad))),
    )


def materialize(recipe: CropRecipe, source_image: Path, *, use_cache: bool = True) -> Path:
    """Produce the crop's pixels, reusing the cache when present."""
    destination = crop_path(recipe)
    if use_cache and destination.is_file():
        return destination

    with Image.open(source_image) as opened:
        image: Image.Image = opened.convert("L") if recipe.grayscale else opened.copy()
        left, top, right, bottom = _padded_box(recipe, image.width, image.height)
        # A degenerate box is legal in the data model (thin glyphs), so widen it by a
        # pixel rather than failing on a zero-size crop.
        right = max(right, left + 1)
        bottom = max(bottom, top + 1)
        crop = image.crop((left, top, right, bottom))

        if recipe.target_height and crop.height and crop.height != recipe.target_height:
            scale = recipe.target_height / crop.height
            crop = crop.resize(
                (max(1, round(crop.width * scale)), recipe.target_height),
                resample=Image.Resampling.LANCZOS,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        buffer = BytesIO()
        crop.save(buffer, format="PNG", optimize=False, compress_level=6)
        destination.write_bytes(buffer.getvalue())
    return destination

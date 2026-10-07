"""Deterministic synthetic document corpus.

Exists so the architecture can be validated end to end with no download, no license
constraints, and no GPU — and so CI exercises the real code path rather than mocks.

The content is chosen for the research question rather than for realism: alongside prose it
carries doses with units, prices, part numbers, dates, and names, because those are the
spans where a *harmful* accepted edit is consequential. A benchmark of only prose would
make overcorrection look cheap.

SYNTHETIC. Nothing measured on this corpus is a research result.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ocr_risk.datasets.base import DatasetAdapter, LicenseSpec, PreflightReport
from ocr_risk.datasets.registry import register_dataset
from ocr_risk.io.hashing import hash_str, sha256_of_bytes
from ocr_risk.io.paths import raw_source_dir
from ocr_risk.io.raw_store import RawStore
from ocr_risk.schemas.base import BBox
from ocr_risk.schemas.documents import DocumentBundle, GtToken, SourceDocument
from ocr_risk.schemas.enums import RedistributionPolicy

__all__ = [
    "SYNTHETIC_LICENSE",
    "SyntheticCorpus",
    "SyntheticDataset",
    "noise_level_of",
    "shingle_hash",
]

DATASET_ID = "synthetic"
GT_POLICY = "synthetic_exact"

SYNTHETIC_LICENSE = LicenseSpec(
    license_id="synthetic-generated",
    name="Generated in-repository; no third-party rights",
    url="",
    redistribution=RedistributionPolicy.ALLOWED,
    attribution_required=False,
    notes="Rendered locally by ocr_risk.datasets.synthetic. Not a real corpus.",
)

_PROSE = [
    "the",
    "patient",
    "received",
    "a",
    "dose",
    "of",
    "the",
    "compound",
    "during",
    "the",
    "second",
    "trial",
    "phase",
    "and",
    "reported",
    "no",
    "adverse",
    "reaction",
    "while",
    "the",
    "sample",
    "remained",
    "under",
    "observation",
    "in",
    "the",
    "laboratory",
    "until",
    "the",
    "following",
    "morning",
    "when",
    "the",
    "analysis",
    "was",
    "completed",
    "by",
    "staff",
]

_UNITS = ("mg", "ug", "ng", "mL", "kg", "IU")
_SURNAMES = (
    "Smith",
    "Okonkwo",
    "Nakamura",
    "Rossi",
    "Dubois",
    "Kowalski",
    "Silva",
    "Andersen",
    "Haddad",
    "Petrov",
    "Nguyen",
    "Fitzgerald",
)
_ALLOYS = ("Ti-6Al-4V", "Al-7075-T6", "Cu-Zn-39", "Fe-18Cr-8Ni", "Mg-AZ31B")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_FAMILIES = ("prose", "dosage", "price", "part_number", "date", "person_name")
_NOISE_LEVELS = ("clean", "light", "moderate")


def _tokens_for_family(rng: Generator, family: str) -> list[str]:
    """Generate one line's worth of tokens for a content family."""
    if family == "dosage":
        out: list[str] = []
        for _ in range(int(rng.integers(2, 4))):
            value = float(rng.choice([0.015, 0.15, 1.5, 2.5, 12.5, 250.0, 500.0]))
            out += [f"{value:g}", str(rng.choice(_UNITS))]
        return ["Dose:", *out]
    if family == "price":
        return [
            "Total:",
            *[
                f"${rng.integers(1, 9999)}.{rng.integers(0, 99):02d}"
                for _ in range(int(rng.integers(2, 4)))
            ],
        ]
    if family == "part_number":
        return [
            "Part:",
            *[str(rng.choice(_ALLOYS)) for _ in range(2)],
            f"REF-{rng.integers(1000, 9999)}-{chr(int(rng.integers(65, 91)))}",
        ]
    if family == "date":
        return [
            "Date:",
            f"{rng.integers(1, 28):02d}",
            str(rng.choice(_MONTHS)),
            str(rng.integers(1990, 2027)),
            f"{rng.integers(0, 23):02d}:{rng.integers(0, 59):02d}",
        ]
    if family == "person_name":
        return ["Signed:", str(rng.choice(_SURNAMES)), str(rng.choice(_SURNAMES))]
    return [str(w) for w in rng.choice(_PROSE, size=int(rng.integers(6, 11)), replace=True)]


def _apply_noise(image: Image.Image, level: str, rng: Generator) -> Image.Image:
    """Degrade a rendered page.

    Degradation is axis-aligned only (no rotation or warp) so recorded token boxes stay
    exact. Ground-truth geometry that drifted from the pixels would corrupt every
    downstream alignment and evidence crop.
    """
    if level == "clean":
        return image
    blur = {"light": 0.4, "moderate": 0.9}[level]
    sigma = {"light": 6.0, "moderate": 16.0}[level]
    out = image.filter(ImageFilter.GaussianBlur(radius=blur))
    array = np.asarray(out, dtype=np.float32)
    noisy = array + rng.normal(0.0, sigma, size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="L")


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    """Parameters of the generated corpus. Hashed into the config, so a change is visible."""

    n_documents: int = 48
    seed: int = 20260817
    page_width: int = 900
    page_height: int = 560
    lines_per_page: tuple[int, int] = (8, 14)
    content_families: tuple[str, ...] = _FAMILIES
    font_sizes: tuple[int, ...] = (16, 18, 20)
    noise_levels: tuple[str, ...] = _NOISE_LEVELS
    margin: int = 40
    line_spacing: int = 14

    def render(self, index: int) -> tuple[Image.Image, list[tuple[str, BBox, int]], str, str]:
        """Render one page.

        Returns the image, ``(token_text, bbox, line_index)`` triples, the linearized
        ground-truth text, and the noise level applied.
        """
        # Seeded per document so a corpus of N and a corpus of N+1 agree on their first
        # N pages; regenerating with a larger n_documents must not perturb existing data.
        rng = default_rng([self.seed, index])
        font_size = int(rng.choice(self.font_sizes))
        noise = str(rng.choice(self.noise_levels))
        font = ImageFont.load_default(size=font_size)

        image = Image.new("L", (self.page_width, self.page_height), color=255)
        draw = ImageDraw.Draw(image)

        placed: list[tuple[str, BBox, int]] = []
        lines: list[str] = []
        y = float(self.margin)
        n_lines = int(rng.integers(self.lines_per_page[0], self.lines_per_page[1] + 1))

        for line_index in range(n_lines):
            family = str(rng.choice(self.content_families))
            tokens = _tokens_for_family(rng, family)
            x = float(self.margin)
            rendered: list[str] = []
            for token in tokens:
                width = draw.textlength(token, font=font)
                if x + width > self.page_width - self.margin:
                    break
                draw.text((x, y), token, font=font, fill=0)
                box = draw.textbbox((x, y), token, font=font)
                placed.append((token, BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]), line_index))
                rendered.append(token)
                x += width + draw.textlength(" ", font=font)
            if rendered:
                lines.append(" ".join(rendered))
            y += font_size + self.line_spacing
            if y > self.page_height - self.margin - font_size:
                break

        return _apply_noise(image, noise, rng), placed, "\n".join(lines), noise


@register_dataset(DATASET_ID)
class SyntheticDataset(DatasetAdapter):
    """Adapter over the generated corpus, materializing it on first use."""

    dataset_id = DATASET_ID
    license = SYNTHETIC_LICENSE
    gt_policy = GT_POLICY

    def __init__(self, **params: object) -> None:
        # YAML gives lists where the dataclass wants tuples (it is frozen and hashable).
        tuple_fields = {f.name for f in fields(SyntheticCorpus) if "tuple" in str(f.type)}
        known = {f.name for f in fields(SyntheticCorpus)}
        unknown = set(params) - known
        if unknown:
            msg = f"unknown synthetic corpus parameters: {sorted(unknown)}"
            raise TypeError(msg)
        kwargs = {
            key: tuple(value) if key in tuple_fields and isinstance(value, list) else value
            for key, value in params.items()
        }
        self.corpus = SyntheticCorpus(**kwargs)  # type: ignore[arg-type]
        self.root = raw_source_dir(DATASET_ID)

    # --- lifecycle ---------------------------------------------------------------
    def preflight(self) -> PreflightReport:
        pages = sorted(self.root.glob("doc-*.png")) if self.root.exists() else []
        if len(pages) < self.corpus.n_documents:
            return PreflightReport(
                dataset_id=DATASET_ID,
                available=False,
                n_documents=len(pages),
                root=self.root,
                problems=[f"found {len(pages)} pages, need {self.corpus.n_documents}"],
                remediation="run: ocr-risk data synth",
            )
        return PreflightReport(
            dataset_id=DATASET_ID,
            available=True,
            n_documents=len(pages),
            root=self.root,
            checksums_verified=True,
        )

    def materialize(self, *, force: bool = False) -> int:
        """Render and persist the corpus into the write-once raw layer.

        Re-running is a no-op when the bytes are identical, which is the intended
        behaviour: the generator is deterministic, so a second call proves it.

        ``force`` deletes the existing pages first, because the raw store refuses to
        change bytes in place. That is safe only because this corpus is *generated* — it
        is reproducible from (seed, parameters) and is not observed evidence. No real
        dataset has an equivalent path; those are downloaded and checksum-verified.
        """
        store = RawStore()
        if force and self.root.exists():
            for stale in sorted(self.root.glob("doc-*")):
                stale.unlink()
        written = 0
        for index in range(self.corpus.n_documents):
            document_id = f"doc-{index:04d}"
            png_path = self.root / f"{document_id}.png"
            gt_path = self.root / f"{document_id}.gt.json"
            if png_path.exists() and gt_path.exists() and not force:
                continue

            image, placed, gt_text, noise = self.corpus.render(index)
            store.write_bytes(png_path, _png_bytes(image))
            store.write_bytes(
                gt_path,
                json.dumps(
                    {
                        "document_id": document_id,
                        "gt_text": gt_text,
                        "noise_level": noise,
                        "width": image.width,
                        "height": image.height,
                        "tokens": [
                            {
                                "text": text,
                                "line": line,
                                "bbox": [box.x0, box.y0, box.x1, box.y1],
                            }
                            for text, box, line in placed
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8"),
            )
            written += 1
        return written

    # --- reading -----------------------------------------------------------------
    def documents(self, limit: int | None = None) -> Iterator[DocumentBundle]:
        self.preflight().raise_if_unavailable()
        # `is not None`, not truthiness: limit=0 means none, not all.
        paths = sorted(self.root.glob("doc-*.png"))[
            : self.corpus.n_documents if limit is None else limit
        ]
        for path in paths:
            yield self._load(path)

    def _load(self, png_path: Path) -> DocumentBundle:
        document_id = png_path.stem
        meta = json.loads((self.root / f"{document_id}.gt.json").read_text(encoding="utf-8"))
        gt_text: str = meta["gt_text"]

        tokens: list[GtToken] = []
        cursor = 0
        for order, entry in enumerate(meta["tokens"]):
            text = entry["text"]
            # Locate each token in the linearized GT so char offsets stay exact even when
            # the same word appears twice on a page.
            start = gt_text.find(text, cursor)
            if start < 0:
                start = cursor
            cursor = start + len(text)
            x0, y0, x1, y1 = entry["bbox"]
            tokens.append(
                GtToken(
                    gt_token_id=f"{document_id}:gt:{order:04d}",
                    document_id=document_id,
                    dataset_id=DATASET_ID,
                    index=order,
                    text=text,
                    char_start=start,
                    char_end=cursor,
                    line_id=f"{document_id}:line:{entry['line']:03d}",
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                )
            )

        document = SourceDocument(
            document_id=document_id,
            dataset_id=DATASET_ID,
            image_path=png_path.relative_to(raw_source_dir(DATASET_ID).parents[2]).as_posix(),
            image_sha256=sha256_of_bytes(png_path.read_bytes()),
            page_index=0,
            width=int(meta["width"]),
            height=int(meta["height"]),
            gt_text=gt_text,
            gt_policy=GT_POLICY,
            has_gt_geometry=True,
            n_gt_tokens=len(tokens),
            license_id=SYNTHETIC_LICENSE.license_id,
            gt_text_shingle_hash=shingle_hash(gt_text),
        )
        return DocumentBundle(document=document, gt_tokens=tuple(tokens))


def _png_bytes(image: Image.Image) -> bytes:
    """Encode deterministically: no timestamp chunk, fixed compression."""
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def shingle_hash(text: str, k: int = 5) -> str:
    """Order-independent hash of a text's character shingles.

    Used for near-duplicate detection: two scans of the same page produce nearly the same
    shingle set even when a few characters differ, so they can be forced into the same
    split bucket (leakage vector L8).
    """
    normalized = " ".join(text.split()).lower()
    if len(normalized) <= k:
        return hash_str(normalized)
    shingles = sorted({normalized[i : i + k] for i in range(len(normalized) - k + 1)})
    return hash_str("␟".join(shingles))


def noise_level_of(document_id: str, root: Path | None = None) -> str:
    """Read back the noise level recorded at render time (used for stratification)."""
    directory = root or raw_source_dir(DATASET_ID)
    meta_path = directory / f"{document_id}.gt.json"
    if not meta_path.is_file():
        return "unknown"
    return str(json.loads(meta_path.read_text(encoding="utf-8")).get("noise_level", "unknown"))

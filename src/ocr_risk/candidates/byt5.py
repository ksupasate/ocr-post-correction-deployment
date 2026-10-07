"""Byte-level seq2seq post-correction, from a real published checkpoint.

Character- and byte-level OCR correction, and ByT5 in particular, are **established prior
art**. This generator is a baseline: it exists so the verifier has stronger proposals to
judge and so the pilot's candidate-quality ceiling can be attributed to the corrector
rather than assumed.

The checkpoint is named, pinned by revision, and downloaded — never simulated. If it is
absent, :meth:`available` returns ``False`` and the study reports the generator as not run.
A byte model that quietly returned nothing would be indistinguishable from a model that
found nothing to fix, which is the confusion this repository's integrity rules exist to
prevent.

**Scope.** ``yelpfeast/byt5-base-english-ocr-correction`` is fine-tuned on English
(wikitext with synthetic OCR noise). It is therefore applicable to the English corpus and
to nothing else: running it on German Fraktur or Indonesian receipts would measure a
language mismatch and report it as generator quality. ``languages`` makes the restriction a
property of the object, and :meth:`propose` returns nothing outside it rather than
producing plausible garbage.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rapidfuzz.distance import Levenshtein

from ocr_risk.candidates.base import BaseGenerator, CandidateProposal, GenerationContext
from ocr_risk.candidates.registry import register_generator

__all__ = ["DEFAULT_BYT5", "Byt5Generator", "Byt5Spec", "project_span"]


def project_span(source: str, rewritten: str, start: int, end: int) -> str | None:
    """Find what ``rewritten`` says in place of ``source[start:end]``.

    A sentence-level corrector rewrites the whole window it is given, but the label and the
    evidence bundle describe **one span**. Returning the window as the candidate would make
    the edit land somewhere other than where it is scored, so the candidate could come out
    "correct" by rewriting text the site does not own.

    The two strings are aligned and the span's character range is carried across: edits
    before the span shift its boundaries, edits inside it are the proposal. Returns
    ``None`` when an edit straddles a boundary — a model that restructured the window did
    not propose an edit to *this* site, and guessing which part belongs here would invent
    an attribution the alignment does not support.
    """
    if not source:
        return None
    left, right = start, end
    for op in Levenshtein.opcodes(source, rewritten):
        if op.tag == "equal":
            continue
        delta = (op.dest_end - op.dest_start) - (op.src_end - op.src_start)
        inserted = rewritten[op.dest_start : op.dest_end]
        if op.src_start == op.src_end == end:
            # A pure insertion at the span's right edge is genuinely ambiguous: it could
            # complete this token or begin the next one. Whitespace decides. "addres" ->
            # "address" appends to the span; "cat" -> "cat dog" does not.
            if inserted.strip() and not inserted[0].isspace():
                right += delta
            continue
        if op.src_start == op.src_end == start:
            # The mirror case, which used to fall through to the "before the span" branch
            # and shift both edges — so a PREFIX repair ("ddress" -> "address") came back
            # unchanged and was reported as the model declining. Same whitespace rule, so
            # a new word inserted before the span still belongs to the left context.
            if inserted.strip() and not inserted[-1].isspace():
                right += delta  # the span's left edge stays; the insertion joins it
            else:
                left += delta
                right += delta
            continue
        if op.src_end <= start:
            left += delta
            right += delta
        elif op.src_start < end:
            if op.src_start < start or op.src_end > end:
                return None
            right += delta
    if left < 0 or right < left or right > len(rewritten):
        return None
    return rewritten[left:right]


@dataclass(frozen=True, slots=True)
class Byt5Spec:
    """A checkpoint, pinned. The revision is part of the identity, not a convenience."""

    repo_id: str
    revision: str
    languages: frozenset[str]
    note: str

    @property
    def identity(self) -> str:
        return f"{self.repo_id}@{self.revision[:12]}"


DEFAULT_BYT5 = Byt5Spec(
    repo_id="yelpfeast/byt5-base-english-ocr-correction",
    revision="19d5c2fd86b87f0a0febb7d2574878a0d68d5294",
    languages=frozenset({"eng"}),
    note=(
        "ByT5-base fine-tuned for English OCR correction on wikitext with synthetic noise. "
        "Public and ungated; the revision is the commit sha read from the model index on "
        "2026-08-19."
    ),
)


@register_generator("byt5")
class Byt5Generator(BaseGenerator):
    """Propose the byte model's rewriting of the span, in its local context."""

    generator_id = "byt5"
    version = "1"

    def __init__(
        self,
        generator_id: str = "byt5",
        repo_id: str = DEFAULT_BYT5.repo_id,
        revision: str = DEFAULT_BYT5.revision,
        languages: Sequence[str] = ("eng",),
        device: str = "auto",
        max_new_tokens: int = 48,
        num_beams: int = 4,
        context_chars: int = 0,
        batch_size: int = 16,
        allow_download: bool | None = None,
    ) -> None:
        self.generator_id = generator_id
        self.spec = Byt5Spec(
            repo_id=repo_id,
            revision=revision,
            languages=frozenset(languages),
            note=DEFAULT_BYT5.note if repo_id == DEFAULT_BYT5.repo_id else "",
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        # Context is off by default. The checkpoint was fine-tuned on whole noisy
        # sentences, so feeding it a span plus neighbours makes it rewrite the neighbours
        # too, and the edit no longer lands on the site the label describes.
        self.context_chars = context_chars
        self.batch_size = batch_size
        self.allow_download = (
            allow_download
            if allow_download is not None
            else os.environ.get("OCR_RISK_ALLOW_MODEL_DOWNLOAD", "0") == "1"
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_error: str = ""

    # ------------------------------------------------------------------ availability

    def available(self) -> bool:
        """Whether the model can actually run here. Never optimistic."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as error:
            self._load_error = f"{error}; install with `uv sync --extra torch`"
            return False
        return self._ensure_loaded()

    def _resolve_device(self) -> str:
        import torch

        if self.device != "auto":
            return self.device
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            kwargs: dict[str, Any] = {"revision": self.spec.revision}
            if not self.allow_download:
                # Offline by default. A study that silently pulls 2 GB the first time it
                # runs is not reproducible from its own record.
                kwargs["local_files_only"] = True
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id, **kwargs)
            # `Any` because transformers' return types vary by version, and pinning this
            # to a stub would make the extra's version a hard constraint on core typing.
            model: Any = T5ForConditionalGeneration.from_pretrained(self.spec.repo_id, **kwargs)
            model.eval()
            self._model = model.to(self._resolve_device())
            torch.set_grad_enabled(False)
        except Exception as error:
            self._load_error = (
                f"{type(error).__name__}: {error}. Set OCR_RISK_ALLOW_MODEL_DOWNLOAD=1 to "
                f"fetch {self.spec.identity}."
            )
            self._model = None
            return False
        return True

    @property
    def load_error(self) -> str:
        return self._load_error

    def fit(self, corpus: Sequence[str]) -> None:
        """No fitting. The checkpoint is frozen, which is what makes it reproducible."""
        return None

    # ------------------------------------------------------------------ generation

    def applies_to(self, language: str) -> bool:
        return language in self.spec.languages

    def _decode(self, texts: Sequence[str]) -> list[str]:
        import torch

        encoded = self._tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self._model.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams
            )
        return [
            str(t).strip()
            for t in self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]

    def propose_batch(
        self, contexts: Sequence[GenerationContext], max_candidates: int = 1
    ) -> list[list[CandidateProposal]]:
        """Rewrite many spans at once.

        Batched because a byte model tokenizes to one token per byte and a per-span call
        wastes most of the device. The per-site :meth:`propose` delegates here so the two
        paths cannot drift.
        """
        if not contexts or not self._ensure_loaded():
            return [[] for _ in contexts]
        windows: list[tuple[str, int, int]] = []
        for c in contexts:
            before = c.context_before[-self.context_chars :] if self.context_chars else ""
            after = c.context_after[: self.context_chars] if self.context_chars else ""
            windows.append(
                (
                    before + c.original_ocr + after,
                    len(before),
                    len(before) + len(c.original_ocr),
                )
            )

        out: list[list[CandidateProposal]] = []
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            decoded = self._decode([window for window, _, _ in batch])
            for context, (window, left, right), rewritten in zip(
                contexts[start : start + self.batch_size], batch, decoded, strict=True
            ):
                proposed = (
                    rewritten
                    if not self.context_chars
                    else project_span(window, rewritten, left, right)
                )
                if not proposed or proposed == context.original_ocr:
                    out.append([])
                    continue
                out.append(
                    [
                        CandidateProposal(
                            text=proposed,
                            score=1.0,
                            metadata=self._meta(model=self.spec.identity, num_beams=self.num_beams),
                        )
                    ][:max_candidates]
                )
        return out

    def propose(self, context: GenerationContext, max_candidates: int) -> list[CandidateProposal]:
        return self.propose_batch([context], max_candidates)[0]

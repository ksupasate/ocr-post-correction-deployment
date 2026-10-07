"""Typed experiment configuration.

Anything that can change a headline number lives here as a validated field with a recorded
default — never as a literal in code. The resolved instance is dumped and canonically
hashed into every run record, so "which settings produced this figure?" is answerable from
the artifact alone.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ocr_risk.schemas.enums import CoverageUnit, HarmPolicy

__all__ = [
    "AlignmentConfig",
    "CalibrationConfig",
    "CandidateConfig",
    "DatasetConfig",
    "EngineConfig",
    "EvidenceConfig",
    "ExperimentConfig",
    "GeneratorSpec",
    "RiskConfig",
    "SiteConfig",
    "SplitConfig",
    "StatsConfig",
    "VerifierSpec",
]


class ConfigModel(BaseModel):
    """Base for configuration sections. Unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DatasetConfig(ConfigModel):
    id: str
    enabled: bool = True
    max_documents: int | None = Field(default=None, ge=1)
    """Cap for development runs. Recorded in the config hash so a capped run can never be
    mistaken for a full one."""
    params: dict[str, Any] = Field(default_factory=dict)


class EngineConfig(ConfigModel):
    id: str
    adapter: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    dataset_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Per-dataset parameter overrides, merged over ``params`` for that corpus only.

    This exists for one reason: recognition language is a property of the *corpus*, not
    of the engine, and the benchmark spans corpora in different languages. Reading German
    Fraktur with an English model produces plausible-looking garbage, so the language has
    to be able to vary per dataset while the engine stays one engine — the leave-one-
    engine-out axis must still see four engines, not eight.

    An override changes the engine fingerprint for that dataset, which is correct: it is
    a different configuration, and the raw store keys on the fingerprint.
    """

    def params_for(self, dataset_id: str) -> dict[str, Any]:
        override = self.dataset_params.get(dataset_id)
        return {**self.params, **override} if override else dict(self.params)


class AlignmentConfig(ConfigModel):
    """Alignment behaviour.

    Every threshold here changes how many correction sites exist and therefore every
    downstream count, so all of them are configured and hashed rather than hardcoded.
    """

    use_geometry: bool = True
    unicode_policy: str = "nfc"
    min_align_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    """Components below this are AMBIGUOUS: retained and counted, excluded from
    evaluation. Lowering it must never be used to manufacture coverage."""
    anchor_min_length: int = Field(default=6, ge=2)
    max_block_chars: int = Field(default=400, ge=16)
    band_margin: int = Field(default=32, ge=1)
    max_anchor_cost: float = Field(default=0.85, ge=0.0, le=1.0)
    substitution_cost: float = Field(default=1.0, gt=0.0)
    insertion_cost: float = Field(default=1.0, gt=0.0)
    deletion_cost: float = Field(default=1.0, gt=0.0)
    confusable_substitution_cost: float = Field(default=0.6, gt=0.0)
    confusable_pairs: tuple[str, ...] = (
        "rn|m",
        "l|1",
        "I|1",
        "O|0",
        "S|5",
        "cl|d",
        "B|8",
        "Z|2",
        "vv|w",
    )
    """Visually confusable pairs given a reduced substitution cost. A modelling choice
    about the *scanner*, not about language, so it is declared as data."""
    weight_char_agreement: float = Field(default=0.5, ge=0.0)
    weight_geometry: float = Field(default=0.3, ge=0.0)
    weight_uniqueness: float = Field(default=0.2, ge=0.0)

    @model_validator(mode="after")
    def _weights_positive(self) -> Self:
        total = self.weight_char_agreement + self.weight_geometry + self.weight_uniqueness
        if total <= 0:
            msg = "alignment confidence weights must sum to a positive value"
            raise ValueError(msg)
        return self


class SiteConfig(ConfigModel):
    include_clean_sites: bool = True
    """Clean sites (``d_before == 0``) must be enumerated: without them overcorrection is
    unobservable. Disabling this is only ever valid for a diagnostic."""
    max_site_chars: int = Field(default=64, ge=1)
    merge_adjacent_components: bool = True


class GeneratorSpec(ConfigModel):
    id: str
    kind: str
    enabled: bool = True
    max_candidates: int = Field(default=4, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class CandidateConfig(ConfigModel):
    generators: tuple[GeneratorSpec, ...] = ()
    max_candidates_per_site: int = Field(default=8, ge=1)
    drop_identity_candidates: bool = True
    """An edit that proposes no change is not an edit; it is filtered from the pool."""
    hard_negatives_in_evaluation: bool = False
    """Whether synthesized adversarial candidates enter the *headline* evaluation.

    Default false, and the distinction is scientific rather than cosmetic. Hard negatives
    are adversarial by construction — around 99% of them are harmful — and they can
    outnumber naturally generated candidates several times over. Pooling them into the
    headline turns the risk-coverage curve into a measurement of "can you reject
    adversarial examples" rather than "how much genuine repair can you safely automate",
    which is the question the project asks.

    They are still used for **fitting**, where teaching the verifier to reject plausible-
    but-unsupported edits is exactly the intended skill, and they are still evaluated —
    as a separately reported challenge set.
    """


class EvidenceConfig(ConfigModel):
    crop_padding_ratio: float = Field(default=0.25, ge=0.0)
    crop_padding_min_px: int = Field(default=4, ge=0)
    crop_target_height: int | None = Field(default=48, ge=8)
    crop_grayscale: bool = True
    context_window_chars: int = Field(default=48, ge=0)
    """Context is taken from the OCR stream only. Taking it from ground truth would leak
    the answer into the features."""


class VerifierSpec(ConfigModel):
    id: str
    kind: str
    evidence_config: str
    """Ablation identifier ``v0``..``v6``; determines which evidence channels survive
    masking before the verifier is constructed."""
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(ConfigModel):
    method: str = "isotonic"
    params: dict[str, Any] = Field(default_factory=dict)
    n_bins: int = Field(default=15, ge=2)
    binning: str = "equal_mass"
    """ECE binning scheme. Equal-width ECE is binning-biased, so both schemes are reported
    and the choice is recorded rather than assumed."""


class RiskConfig(ConfigModel):
    epsilon_grid: tuple[float, ...] = (0.01, 0.005, 0.001)
    """Harmful-edit tolerances. Configurable, never a hardcoded scientific constant."""
    primary_verifier: str = "v6_full"
    """The evidence configuration H1's verdict is read from.

    Named because the pre-registration was silent on it, and silence is not neutral: the
    gate previously took `.any()` across all seven configurations, which gave each engine
    seven uncorrected chances to register degradation and — on this pilot — changed the
    verdict. `v6_full` is the configuration with every evidence channel, the project's
    proposed method by construction rather than by outcome. Every configuration's result
    is reported alongside."""
    primary_epsilon: float = Field(default=0.01, gt=0.0, lt=1.0)
    """The tolerance the gate judges at, named rather than derived.

    Taking max(epsilon_grid) would silently pick the most permissive level -- the one
    most favourable to a positive verdict -- and nobody would have chosen it. Fixed in
    docs/preregistration.md before the results were produced."""
    delta: float = Field(default=0.1, gt=0.0, lt=1.0)
    """Failure probability for the distribution-free risk bound."""
    controller: str = "ltt_bentkus"
    coverage_unit: CoverageUnit = CoverageUnit.SITE
    harm_policy: HarmPolicy = HarmPolicy.STRICT_WORSENING
    sensitivity_harm_policies: tuple[HarmPolicy, ...] = (
        HarmPolicy.STRICT_WORSENING,
        HarmPolicy.NON_IMPROVING,
        HarmPolicy.EXACT_ONLY,
    )
    """All three are reported. The primary policy must not be privileged silently."""
    site_policy: str = "argmax_above_tau"
    n_threshold_grid: int = Field(default=200, ge=2)

    @model_validator(mode="after")
    def _epsilons_valid(self) -> Self:
        if not self.epsilon_grid:
            msg = "risk.epsilon_grid must not be empty"
            raise ValueError(msg)
        if any(not 0.0 < e < 1.0 for e in self.epsilon_grid):
            msg = f"risk.epsilon_grid values must lie in (0, 1); got {self.epsilon_grid}"
            raise ValueError(msg)
        if self.harm_policy not in self.sensitivity_harm_policies:
            msg = "risk.harm_policy must be included in risk.sensitivity_harm_policies"
            raise ValueError(msg)
        return self


class SplitConfig(ConfigModel):
    """The two split axes. Both are always applied; see docs/cross_engine_protocol.md."""

    partition_id: str = ""
    """Identity of the frozen document partition, shared across experiments.

    Empty means "use the experiment name". Two experiments that must be compared -- the
    zero-shot arm and the engine-aware reference arm of H1 -- have to draw on the SAME
    partition, or the contrast between them is confounded by which documents happened to
    be evaluated. Keying the frozen partition on the experiment name alone would give
    them different ones while every run record still looked self-consistent."""
    protocol: str = "loeo_zero_shot"
    partition_seed: int = 20260817
    fit_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    calibrate_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify_by: tuple[str, ...] = ("dataset_id", "noise_tertile")
    few_shot_k: tuple[int, ...] = (5, 10, 25, 50)
    few_shot_repeats: int = Field(default=20, ge=1)
    allow_document_overlap: bool = False
    """Deliberate contamination for the ``doc_overlap`` diagnostic only. Setting this true
    marks the run ``leaky`` and bars it from every headline table."""
    selection_scope: str = "fit_only"

    @model_validator(mode="after")
    def _fractions_leave_test_room(self) -> Self:
        used = self.fit_fraction + self.calibrate_fraction
        if used >= 1.0:
            msg = (
                f"fit_fraction + calibrate_fraction = {used:.3f} leaves no documents for "
                "the evaluation split"
            )
            raise ValueError(msg)
        return self

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.fit_fraction - self.calibrate_fraction


class StatsConfig(ConfigModel):
    n_bootstrap: int = Field(default=10_000, ge=100)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    bootstrap_unit: str = "document"
    """Documents, not tokens: tokens within a page are correlated, so token-level
    resampling would produce dishonestly narrow intervals."""
    multiplicity: str = "holm"
    bootstrap_seed: int = 7


class ExperimentConfig(ConfigModel):
    """A complete, resolved experiment specification."""

    name: str
    description: str = ""
    seed: int = 20260817
    synthetic: bool = False
    """Set by synthetic experiment configs; propagates into artifacts and figures."""
    datasets: tuple[DatasetConfig, ...] = ()
    engines: tuple[EngineConfig, ...] = ()
    alignment: AlignmentConfig = AlignmentConfig()
    sites: SiteConfig = SiteConfig()
    candidates: CandidateConfig = CandidateConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    verifiers: tuple[VerifierSpec, ...] = ()
    calibration: CalibrationConfig = CalibrationConfig()
    risk: RiskConfig = RiskConfig()
    splits: SplitConfig = SplitConfig()
    stats: StatsConfig = StatsConfig()

    @model_validator(mode="after")
    def _primary_epsilon_is_on_the_grid(self) -> Self:
        if self.risk.primary_epsilon not in self.risk.epsilon_grid:
            msg = (
                f"primary_epsilon={self.risk.primary_epsilon} is not in the reporting grid "
                f"{list(self.risk.epsilon_grid)}. The gate must judge at a tolerance the "
                "study actually reports."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _protocol_needs_enough_engines(self) -> Self:
        enabled = [e.id for e in self.engines if e.enabled]
        if len(enabled) != len(set(enabled)):
            msg = f"duplicate engine ids in configuration: {enabled}"
            raise ValueError(msg)
        if self.splits.protocol.startswith("loeo") and len(enabled) < 2:
            msg = (
                f"protocol {self.splits.protocol!r} holds one engine out, so it needs at "
                f"least 2 enabled engines; got {enabled}"
            )
            raise ValueError(msg)
        return self

    @property
    def enabled_engine_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.engines if e.enabled)

    @property
    def enabled_dataset_ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.datasets if d.enabled)

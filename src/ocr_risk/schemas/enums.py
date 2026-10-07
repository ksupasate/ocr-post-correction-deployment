"""Closed vocabularies for the canonical data model.

Every enum here is a scientific commitment, not a convenience. Adding a member changes
what the benchmark can express; changing a value breaks stored artifacts. Both require a
``SCHEMA_VERSION`` bump in :mod:`ocr_risk.schemas`.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AlignmentRelation",
    "AlignmentStatus",
    "CandidatePool",
    "CoverageUnit",
    "DecisionAction",
    "EvidenceField",
    "HarmPolicy",
    "OutcomeIfAccepted",
    "OutcomeIfRejected",
    "RedistributionPolicy",
    "SiteKind",
    "SplitRole",
    "StageName",
    "harmful_outcomes",
]


class SplitRole(StrEnum):
    """Why a row is being read. Determines which engines and documents are reachable.

    ``FIT`` and ``CALIBRATE`` expose labels to fitting code; ``EVALUATE`` does not
    (labels for evaluation flow through :mod:`ocr_risk.metrics` only). ``DEV`` is an
    inner split carved out of the fit scope for model selection.
    """

    FIT = "fit"
    DEV = "dev"
    CALIBRATE = "calibrate"
    EVALUATE = "evaluate"


class AlignmentRelation(StrEnum):
    """Shape of an OCR-span to ground-truth-token correspondence."""

    ONE_TO_ONE = "one_to_one"
    SPLIT = "split"
    """N OCR spans map to 1 GT token: the engine split a word."""
    MERGE = "merge"
    """1 OCR span maps to N GT tokens: the engine merged words."""
    MANY_TO_MANY = "many_to_many"
    OCR_INSERTION = "ocr_insertion"
    """OCR produced text with no GT counterpart (hallucination)."""
    OCR_DELETION = "ocr_deletion"
    """GT text has no OCR counterpart (omission)."""


class AlignmentStatus(StrEnum):
    """Trust level of an alignment component.

    Only ``RESOLVED`` components may produce evaluated correction sites. The others are
    retained in the artifact with a reason code so ambiguity is measured, never hidden.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    """Below the confidence floor. Kept and counted, excluded from labeling/evaluation."""
    UNRESOLVED = "unresolved"
    """No acceptable alignment found. Never force a match to raise coverage."""
    OUT_OF_REGION = "out_of_region"
    """Fell outside every anchored region during geometric anchoring."""


class SiteKind(StrEnum):
    """What kind of discrepancy (if any) a correction site represents."""

    CLEAN = "clean"
    """``d_before == 0``. Retained deliberately: without clean sites, overcorrection is
    unobservable."""
    SUBSTITUTION = "substitution"
    SEGMENTATION = "segmentation"
    INSERTION = "insertion"
    DELETION = "deletion"
    EXCLUDED = "excluded"
    """Built from ambiguous/unresolved components; carried but not evaluated."""


class AnchorKind(StrEnum):
    """How a discovered site is anchored in the OCR artifacts alone (CGV3).

    An anchor is reproducible from OCR output by construction: no kind depends on a
    ground-truth token, alignment, or distance. The ``GAP`` kind is the structural
    answer to R-65 -- an insertion-repair site anchored on the observable boundary
    between two adjacent spans, never on a GT-only component pinning ``[0, 0]``.
    """

    TOKEN = "token"
    """One OCR span, by its stream range."""
    TOKEN_PAIR = "token_pair"
    """Two adjacent same-line spans and the exact stream slice between them."""
    GAP = "gap"
    """The boundary between two adjacent spans: ``(left.end, right.start)``."""
    TOKEN_INTERIOR = "token_interior"
    """A character offset inside one span (within-token splits, hyphenation)."""


class DiscoveryProvenance(StrEnum):
    """Why a discovered site exists -- the OCR signal that fired, retained per site."""

    LOW_CONFIDENCE_TOKEN = "low_confidence_token"
    LEXICAL_ANOMALY = "lexical_anomaly"
    POSSIBLE_SPLIT = "possible_split"
    POSSIBLE_MERGE = "possible_merge"
    GAP_ANOMALY = "gap_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    BOUNDARY_ANOMALY = "boundary_anomaly"


class OutcomeIfAccepted(StrEnum):
    """Counterfactual outcome of accepting a candidate edit.

    With ``d_before = lev(O, G)`` and ``d_after = lev(Y, G)`` in raw characters:

    ========================================  ===========================
    condition                                 member
    ========================================  ===========================
    ``Y == O``                                ``IDENTITY``
    ``d_before == 0`` and ``d_after > 0``     ``OVERCORRECTION``   (harmful)
    ``d_before > 0`` and ``d_after == 0``     ``TRUE_CORRECTION``
    ``0 < d_after < d_before``                ``PARTIAL_IMPROVEMENT``
    ``d_after == d_before`` and ``Y != O``    ``LATERAL_CHANGE``
    ``d_before > 0`` and ``d_after > d_before``  ``MISCORRECTION`` (harmful)
    ========================================  ===========================
    """

    IDENTITY = "identity"
    TRUE_CORRECTION = "true_correction"
    PARTIAL_IMPROVEMENT = "partial_improvement"
    LATERAL_CHANGE = "lateral_change"
    MISCORRECTION = "miscorrection"
    OVERCORRECTION = "overcorrection"


class OutcomeIfRejected(StrEnum):
    """Counterfactual outcome of preserving the original OCR at a site."""

    PRESERVATION = "preservation"
    """``d_before == 0``: correctly left alone."""
    MISSED_ERROR = "missed_error"
    """``d_before > 0``: a real error was not repaired (opportunity cost, not harm)."""


class CandidatePool(StrEnum):
    """Which population a candidate belongs to. Not interchangeable, ever.

    ``NATURAL`` is what a generator emitted from OCR alone: the population a deployed
    system would face, and the only one a headline coverage or risk number may be computed
    over. ``CHALLENGE`` is synthesized adversarially — around 99% harmful by construction,
    and able to outnumber the natural pool several times over. Pooling them turns a
    risk-coverage curve into a measurement of adversarial rejection rather than of safe
    repair, which is a different question with a much more flattering answer.

    The distinction is a schema field rather than a convention because it was previously
    carried by a boolean named for its *cause* (``is_synthetic_hard_negative``) rather than
    its *meaning*, and a reader had to know that the boolean was what kept the denominators
    apart.
    """

    NATURAL = "natural"
    CHALLENGE = "challenge"


class HarmPolicy(StrEnum):
    """Which accepted outcomes count as harmful.

    This choice materially moves the headline numbers, so it is a configured policy with
    a mandatory sensitivity analysis over all three variants, never a hardcoded constant.
    """

    STRICT_WORSENING = "strict_worsening"
    """Default. Harmful iff the edit strictly increased distance to ground truth."""
    NON_IMPROVING = "non_improving"
    """Also counts lateral changes (text churn with no gain) as harmful."""
    EXACT_ONLY = "exact_only"
    """Most conservative: only an exact repair is beneficial; every other change harms."""


class CoverageUnit(StrEnum):
    """Denominator for coverage.

    ``SITE`` is the default: sites are the decision points, whereas candidates-per-site is
    an artifact of the generator and would make generators incomparable.
    """

    SITE = "site"
    CANDIDATE = "candidate"


class DecisionAction(StrEnum):
    """Action taken at a correction site by the decision controller."""

    PRESERVE = "preserve"
    CORRECT = "correct"


class EvidenceField(StrEnum):
    """Evidence channels a verifier may consume.

    Ablation configurations V0-V6 are expressed as subsets of these. Masking removes the
    field before the verifier is constructed, so an ablation cannot leak by oversight.
    """

    ORIGINAL_OCR = "original_ocr"
    CANDIDATE_TEXT = "candidate_text"
    IMAGE_CROP = "image_crop"
    TEXT_CONTEXT = "text_context"
    OCR_CONFIDENCE = "ocr_confidence"
    SPATIAL = "spatial"


class RedistributionPolicy(StrEnum):
    """Whether a dataset's license permits shipping its files from this repository."""

    ALLOWED = "allowed"
    PROHIBITED = "prohibited"
    UNCLEAR = "unclear"
    """Treated as prohibited by the license audit; recorded distinctly for honesty."""


class StageName(StrEnum):
    """The 15 pipeline stages. Order defines the provenance DAG's topology."""

    ACQUIRE = "acquire"
    MANIFEST = "manifest"
    OCR = "ocr"
    CANONICALIZE = "canonicalize"
    ALIGN = "align"
    SITES = "sites"
    CANDIDATES = "candidates"
    LABELS = "labels"
    EVIDENCE = "evidence"
    PREDICT = "predict"
    CALIBRATE = "calibrate"
    DECIDE = "decide"
    EVALUATE = "evaluate"
    REPORT = "report"
    GATE = "gate"


_HARMFUL_BY_POLICY: dict[HarmPolicy, frozenset[OutcomeIfAccepted]] = {
    HarmPolicy.STRICT_WORSENING: frozenset(
        {OutcomeIfAccepted.MISCORRECTION, OutcomeIfAccepted.OVERCORRECTION}
    ),
    HarmPolicy.NON_IMPROVING: frozenset(
        {
            OutcomeIfAccepted.MISCORRECTION,
            OutcomeIfAccepted.OVERCORRECTION,
            OutcomeIfAccepted.LATERAL_CHANGE,
        }
    ),
    HarmPolicy.EXACT_ONLY: frozenset(
        {
            OutcomeIfAccepted.MISCORRECTION,
            OutcomeIfAccepted.OVERCORRECTION,
            OutcomeIfAccepted.LATERAL_CHANGE,
            OutcomeIfAccepted.PARTIAL_IMPROVEMENT,
        }
    ),
}


def harmful_outcomes(policy: HarmPolicy) -> frozenset[OutcomeIfAccepted]:
    """Return the accepted-outcome members that ``policy`` treats as harmful.

    ``IDENTITY`` is never harmful under any policy: an identity edit proposes no change
    and is filtered out of the candidate pool upstream.
    """
    return _HARMFUL_BY_POLICY[policy]

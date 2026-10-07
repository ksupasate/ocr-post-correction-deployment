"""Walking the artifact provenance DAG.

Answers the question every reviewer eventually asks: *where did this number come from?*
Starting at a figure or a metrics table, follow ``inputs`` upward until reaching the source
manifest, verifying each recorded hash on the way.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from ocr_risk.schemas.enums import StageName
from ocr_risk.schemas.run_record import RunRecord

__all__ = ["LineageNode", "LineageReport", "RecordSource", "build_lineage", "render_lineage"]


class RecordSource(Protocol):
    """The slice of an artifact store this walker needs.

    Typing against a protocol instead of importing ``io.artifacts`` keeps the layer edge
    one-directional (``io.artifacts`` builds on ``provenance``, never the reverse) and
    lets the walker be tested against a fake store with no filesystem.
    """

    def iter_records(self, stage: StageName | None = ...) -> Iterator[RunRecord]: ...

    def verify_outputs(self, record: RunRecord) -> bool: ...


@dataclass(frozen=True, slots=True)
class LineageNode:
    """One run in the lineage, with the integrity of its outputs checked."""

    record: RunRecord
    depth: int
    outputs_intact: bool


@dataclass(slots=True)
class LineageReport:
    """Result of walking upward from one artifact."""

    root_run_id: str
    nodes: list[LineageNode] = field(default_factory=list)
    missing_runs: list[str] = field(default_factory=list)
    """Input run_ids that are referenced but have no finalized record. A non-empty list
    means the chain is broken: an upstream artifact was deleted or never completed."""
    corrupted_runs: list[str] = field(default_factory=list)
    """Runs whose files no longer hash to what the record claims."""

    @property
    def complete(self) -> bool:
        return not self.missing_runs and not self.corrupted_runs

    @property
    def synthetic(self) -> bool:
        return any(node.record.synthetic for node in self.nodes)

    @property
    def leaky(self) -> bool:
        return any(node.record.leaky for node in self.nodes)

    def summary(self) -> dict[str, object]:
        return {
            "root_run_id": self.root_run_id,
            "n_runs": len(self.nodes),
            "stages": sorted({n.record.stage.value for n in self.nodes}),
            "complete": self.complete,
            "synthetic": self.synthetic,
            "leaky": self.leaky,
            "missing_runs": self.missing_runs,
            "corrupted_runs": self.corrupted_runs,
        }


def build_lineage(store: RecordSource, run_id: str) -> LineageReport:
    """Breadth-first walk from ``run_id`` through every upstream run."""
    by_id = {record.run_id: record for record in store.iter_records()}
    report = LineageReport(root_run_id=run_id)

    if run_id not in by_id:
        report.missing_runs.append(run_id)
        return report

    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(run_id, 0)])
    while queue:
        current_id, depth = queue.popleft()
        if current_id in seen:
            continue
        seen.add(current_id)

        record = by_id.get(current_id)
        if record is None:
            report.missing_runs.append(current_id)
            continue

        intact = store.verify_outputs(record)
        if not intact:
            report.corrupted_runs.append(current_id)
        report.nodes.append(LineageNode(record=record, depth=depth, outputs_intact=intact))

        for ref in record.inputs:
            if ref.run_id not in seen:
                queue.append((ref.run_id, depth + 1))

    report.nodes.sort(key=lambda n: (n.depth, n.record.run_id))
    return report


def render_lineage(report: LineageReport) -> str:
    """Human-readable lineage tree."""
    lines = [f"lineage of {report.root_run_id}"]
    for node in report.nodes:
        marker = "ok " if node.outputs_intact else "BAD"
        flags = []
        if node.record.synthetic:
            flags.append("synthetic")
        if node.record.leaky:
            flags.append("LEAKY")
        if node.record.git.dirty:
            flags.append("dirty-tree")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        indent = "  " * node.depth
        lines.append(
            f"  {marker} {indent}{node.record.stage.value:<12} {node.record.run_id}"
            f"  ({len(node.record.outputs)} outputs){suffix}"
        )
    for missing in report.missing_runs:
        lines.append(f"  MISSING upstream run: {missing}")
    for corrupt in report.corrupted_runs:
        lines.append(f"  CORRUPTED outputs: {corrupt}")
    lines.append(f"  complete={report.complete} synthetic={report.synthetic} leaky={report.leaky}")
    return "\n".join(lines)

"""Schema-enforced Parquet IO for the canonical tables.

Reads and writes always go through a :class:`~ocr_risk.schemas.TableSpec`. Inferring a
schema from data would let an all-null column change type between runs and turn a stage
boundary into a silent coercion; declaring it makes that a load-time error instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ocr_risk.schemas.arrow import TableSpec, records_to_table, table_to_records

__all__ = ["read_records", "read_table", "row_count", "write_records", "write_table"]

# zstd at a moderate level: these tables are read far more often than written, and the
# artifact tree lives on a nearly full disk.
_COMPRESSION = "zstd"
_COMPRESSION_LEVEL = 6


def write_table(path: Path, spec: TableSpec, table: pa.Table) -> Path:
    """Write an Arrow table, rejecting any deviation from the declared schema."""
    if table.schema != spec.schema:
        msg = f"table {spec.name!r} does not match its declared schema; refusing to write"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"ocr_risk.table": spec.name.encode(),
        b"ocr_risk.table_version": str(spec.version).encode(),
    }
    schema_with_meta = table.schema.with_metadata({**(table.schema.metadata or {}), **metadata})
    pq.write_table(
        table.replace_schema_metadata(schema_with_meta.metadata),
        path,
        compression=_COMPRESSION,
        compression_level=_COMPRESSION_LEVEL,
        # Deterministic output: no timestamps or writer-specific statistics that would make
        # two identical runs produce different bytes.
        store_schema=True,
        write_statistics=False,
    )
    return path


def write_records(path: Path, spec: TableSpec, records: list[Any]) -> Path:
    return write_table(path, spec, records_to_table(spec, records))


def _backfill_defaulted_columns(spec: TableSpec, table: pa.Table) -> pa.Table:
    """Add columns the declaration gained since this artifact was written.

    Only for fields whose pydantic model supplies a default, and only by *adding* — never
    by dropping, reordering into a different type, or changing an existing column. Anything
    else is a genuine mismatch and still raises.

    Without this, adding one optional field orphans every artifact ever written, and this
    repository's artifacts are supposed to stay readable years later: the H1 pilot's
    candidates table predates the ``pool`` column and is immutable by policy, so
    regenerating it is not an option and reading it is a requirement.
    """
    missing = [name for name in spec.schema.names if name not in table.schema.names]
    if not missing:
        return table
    defaults = {
        name: field.get_default(call_default_factory=True)
        for name, field in spec.model.model_fields.items()
        if not field.is_required()
    }
    unfillable = [name for name in missing if name not in defaults]
    if unfillable:
        return table  # let the caller's schema comparison report it
    for name in missing:
        value = defaults[name]
        # StrEnum and friends round-trip through the arrow column as their value.
        filled = value.value if hasattr(value, "value") else value
        field = spec.schema.field(name)
        table = table.append_column(field, pa.array([filled] * table.num_rows, type=field.type))
    return table.select(spec.schema.names)


def read_table(path: Path, spec: TableSpec) -> pa.Table:
    """Read a Parquet file and assert it is the table it claims to be."""
    table = pq.read_table(path)
    stored = (table.schema.metadata or {}).get(b"ocr_risk.table")
    if stored is not None and stored.decode() != spec.name:
        msg = f"{path} holds table {stored.decode()!r}, not {spec.name!r}"
        raise ValueError(msg)
    bare = _backfill_defaulted_columns(spec, table.replace_schema_metadata(None))
    if bare.schema != spec.schema:
        msg = (
            f"{path}: stored schema does not match the current declaration for "
            f"{spec.name!r}. Regenerate the artifact or bump SCHEMA_VERSION.\n"
            f"expected: {spec.schema}\nactual:   {bare.schema}"
        )
        raise ValueError(msg)
    return bare


def read_records(path: Path, spec: TableSpec) -> list[Any]:
    """Read and re-validate every row through the pydantic model."""
    return table_to_records(spec, read_table(path, spec))


def read_dataframe(path: Path, spec: TableSpec) -> pd.DataFrame:
    """Read as a pandas frame for vectorized analysis.

    Note for callers under ``verify/``, ``calibrate/``, and ``risk/``: fitting code must
    obtain frames from ``SplitPlan.view(...)``, never from this function directly.
    """
    frame: pd.DataFrame = read_table(path, spec).to_pandas()
    return frame


def row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)

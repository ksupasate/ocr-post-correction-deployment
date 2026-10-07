"""The software archive must exclude corpus payloads and legacy scientific results."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ACCESS_RECORDS = {
    "results/generated/sgv1/reserve/pre_access_freshness_snapshot.json",
    "results/generated/sgv1/reserve/reserve_lock_red_team.json",
}


def test_software_tree_contains_real_implementation_and_no_legacy_scientific_payload():
    assert (ROOT / "src/ocr_risk/metrics/discrimination.py").is_file()
    files = {str(p.relative_to(ROOT)) for p in (ROOT / "results").rglob("*") if p.is_file()}
    assert files >= ALLOWED_ACCESS_RECORDS
    assert all(
        name in ALLOWED_ACCESS_RECORDS
        or name.startswith("results/paper3_vf2/run1/")
        or name.startswith("results/paper3_vf2/preparation/")
        or name == "results/paper3_vf2/P3_VF2_FINAL_DELIVERY_MANIFEST.json"
        for name in files
    )


def test_software_tree_omits_corpus_and_model_payloads():
    assert not list((ROOT / "data/raw").rglob("*"))
    assert not list((ROOT / "data/processed").rglob("*.parquet"))
    assert not list((ROOT / "artifacts").rglob("*.pkl"))

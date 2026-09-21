"""Validazioni diagnostiche CAD: i limiti di accuratezza restano nel report."""

import pytest

from physical_ai_mujoco.evaluation.sensor_validation import validate_cad


def test_cad_partial_surface_report_has_all_exposures_and_24_poses():
    report = validate_cad()
    rows = report["rows"]
    assert len(rows) == 120
    assert {row["pose"] for row in rows} == set(range(24))
    assert {row["exposure_fraction"] for row in rows} == {0.1, 0.3, 0.5, 0.7, 0.9}
    assert all(row["observed_points"] >= 6 for row in rows)
    assert all("accepted" in row for row in rows)
    assert all("raw_rotation_error_deg" in row for row in rows if row["accepted"])
    assert all(row["size_ratio"] == pytest.approx([1, 1, 1]) for row in rows if row["accepted"])

"""Tests for the quality_control.json collector."""

import tempfile
import unittest
from pathlib import Path

from aind_data_schema.core.quality_control import QualityControl, Status

from aind_ophys_motion_correction_library import qc


class TestQCMetrics(unittest.TestCase):
    """Metric builders carry the standard tags and grouping."""

    def test_registration_summary_metric(self) -> None:
        """Registration-summary metric names, tags, and reference set."""
        m = qc.build_registration_summary_metric("plane_0", "ref.png")
        self.assertIn("plane_0", m.name)
        self.assertEqual(m.tags["evaluation"], "Registration Summary")
        self.assertEqual(m.reference, "ref.png")

    def test_fov_quality_metric(self) -> None:
        """FOV-quality metric carries the evaluation + type tags."""
        m = qc.build_fov_quality_metric("plane_0", "combined.png")
        self.assertEqual(m.tags["evaluation"], "FOV Quality")
        self.assertEqual(m.tags["type"], "Operational QC")

    def test_pending_status(self) -> None:
        """The default status history entry is PENDING."""
        self.assertEqual(qc.pending_qc_status().status, Status.PENDING)

    def test_dropdowns_ship_empty_so_review_is_requested(self) -> None:
        """Both dropdowns leave the value unset.

        qcportal treats "" as pending; a preselected option renders as
        already answered, so a reviewer skips it. In both of these the
        passing option is index 0, so a preset value auto-passed the very
        review the metric exists to request.
        """
        for m in (
            qc.build_registration_summary_metric("plane_0", "a.png"),
            qc.build_fov_quality_metric("plane_0", "b.png"),
        ):
            self.assertEqual(m.value.value, "")
            self.assertEqual(
                m.status_history[0].status, Status.PENDING
            )


class TestQualityControlCollector(unittest.TestCase):
    """build_quality_control + write round-trip."""

    def test_build_and_write(self) -> None:
        """Metrics collect into a QualityControl that re-validates."""
        d = Path(tempfile.mkdtemp())
        metrics = [
            qc.build_registration_summary_metric("plane_0", "a.png"),
            qc.build_fov_quality_metric("plane_0", "b.png"),
        ]
        q = qc.build_quality_control(metrics)
        self.assertEqual(q.default_grouping, ["evaluation"])
        self.assertEqual(len(q.metrics), 2)
        out = qc.write_quality_control_json(q, d)
        self.assertEqual(out.name, "quality_control.json")
        QualityControl.model_validate_json(out.read_text())


if __name__ == "__main__":
    unittest.main()

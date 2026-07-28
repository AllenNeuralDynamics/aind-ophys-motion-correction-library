"""Tests for the v1/v2 metadata read layer + processing.json builder.

The v1/v2 field getters live in the shared ``aind_pophys_metadata`` library
(tested there); these tests cover the motion-correction surface:
``load_motion_correction_metadata`` dispatch + the DataProcess/Processing
builders.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.core.processing import Processing
from aind_pophys_metadata import io, processing as shared_processing

from aind_ophys_motion_correction_library.utils import metadata_utils as mu

_PIPELINE_ENV = {
    shared_processing.PIPELINE_NAME_ENV: "aind-pophys-pipeline",
    shared_processing.PIPELINE_VERSION_ENV: "2.0.0",
    shared_processing.PIPELINE_URL_ENV: "https://example.com/pipeline",
}


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write a JSON payload into directory/name."""
    (directory / name).write_text(json.dumps(payload))


def _imaging_config_type() -> str:
    """Return the v2 ImagingConfig discriminator string."""
    from aind_data_schema.components.configs import ImagingConfig

    return io.object_type_value(ImagingConfig)


class TestLoadMetadata(unittest.TestCase):
    """File-presence dispatch and independent id reads."""

    def test_v2_dispatch(self):
        """acquisition.json present -> v2 path with instrument_id."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "acquisition.json",
            {
                "instrument_id": "MESO.1",
                "data_streams": [
                    {
                        "configurations": [
                            {
                                "object_type": _imaging_config_type(),
                                "sampling_strategy": {"frame_rate": 9.6},
                            }
                        ]
                    }
                ],
            },
        )
        _write(d, "data_description.json", {"name": "mouse_1_2024"})
        _write(d, "subject.json", {"subject_id": "781"})
        md = mu.load_motion_correction_metadata(d)
        self.assertEqual(md.frame_rate, 9.6)
        self.assertEqual(md.instrument_id, "MESO.1")
        self.assertEqual(md.subject_id, "781")
        self.assertEqual(md.dataset_name, "mouse_1_2024")

    def test_v1_dispatch(self):
        """session.json present -> v1 path with rig_id as instrument."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "session.json",
            {
                "rig_id": "Bergamo.1",
                "data_streams": [{"ophys_fovs": [{"frame_rate": 11.0}]}],
            },
        )
        md = mu.load_motion_correction_metadata(d)
        self.assertEqual(md.frame_rate, 11.0)
        self.assertEqual(md.instrument_id, "Bergamo.1")
        self.assertIsNone(md.subject_id)

    def test_subject_id_canonical_from_core(self):
        """subject_id is read from the core file when present."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "session.json",
            {
                "rig_id": "MESO.2",
                "subject_id": "555",
                "data_streams": [{"ophys_fovs": [{"frame_rate": 5.0}]}],
            },
        )
        _write(d, "subject.json", {"subject_id": "999"})
        md = mu.load_motion_correction_metadata(d)
        self.assertEqual(md.subject_id, "555")

    def test_platform_fallback_used(self):
        """frame_rate falls back to platform.json when core lacks it."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "session.json",
            {"rig_id": "MESO.2", "data_streams": [{"ophys_fovs": [{}]}]},
        )
        _write(
            d,
            "platform.json",
            {"imaging_plane_groups": [{"acquisition_framerate_Hz": 30.9}]},
        )
        md = mu.load_motion_correction_metadata(d)
        self.assertEqual(md.frame_rate, 30.9)

    def test_raises_without_core_file(self):
        """Neither core file present -> FileNotFoundError."""
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            mu.load_motion_correction_metadata(d)


class TestProcessingBuilders(unittest.TestCase):
    """DataProcess / Processing construction and serialization."""

    def test_build_and_write_round_trip(self):
        """A built Processing writes and re-validates as a v2 object."""
        d = Path(tempfile.mkdtemp())
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process(
            parameters={"a": 1},
            start_time=now,
            end_time=now,
            input_data=["raw_asset"],
            output_path="plane_0/motion_correction",
        )
        proc = mu.build_processing([dp])
        out = mu.write_processing_json(proc, d)
        self.assertEqual(out.name, "processing.json")
        Processing.model_validate_json(out.read_text())

    def test_input_data_optional(self):
        """input_data=None yields a Code without input assets."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process({}, now, now)
        self.assertIsNone(dp.code.input_data)

    def test_resources_recorded(self):
        """Host resource usage is recorded on the process."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process({}, now, now)
        self.assertIsNotNone(dp.resources)

    def test_experimenters_defaults_empty(self):
        """experimenters is an empty list of people, never an institution."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process({}, now, now)
        self.assertEqual(dp.experimenters, [])

    def test_error_recorded_in_notes(self):
        """A run error is summarized in DataProcess.notes."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process(
            {}, now, now, error=RuntimeError("suite2p blew up")
        )
        self.assertIn("RuntimeError", dp.notes)
        self.assertIn("suite2p blew up", dp.notes)

    def test_notes_none_on_success(self):
        """No error -> no notes."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process({}, now, now)
        self.assertIsNone(dp.notes)

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_pipeline_identity_from_env(self):
        """In a pipeline run, the process links to the pipeline."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process({}, now, now)
        doc = mu.build_processing([dp])
        self.assertEqual(dp.pipeline_name, "aind-pophys-pipeline")
        self.assertEqual(doc.pipelines[0].name, "aind-pophys-pipeline")

    @patch.dict(os.environ, {}, clear=True)
    def test_no_pipeline_identity_standalone(self):
        """Standalone runs omit pipeline linkage rather than inventing it."""
        now = datetime.now(timezone.utc)
        dp = mu.build_data_process({}, now, now)
        doc = mu.build_processing([dp])
        self.assertIsNone(dp.pipeline_name)
        self.assertIsNone(doc.pipelines)


if __name__ == "__main__":
    unittest.main()

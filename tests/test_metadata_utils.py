"""Tests for the v1/v2 metadata read layer + processing.json builder."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from aind_data_schema.core.processing import Processing

from aind_ophys_motion_correction_library.utils import metadata_utils as mu


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write a JSON payload into directory/name."""
    (directory / name).write_text(json.dumps(payload))


class TestFrameRateGetters(unittest.TestCase):
    """Unit tests for the individual frame-rate getters."""

    def test_v1_first_fov_with_rate(self):
        """v1 returns the first ophys_fov frame_rate, coerced to float."""
        session = {
            "data_streams": [
                {"ophys_fovs": [{}]},
                {"ophys_fovs": [{"frame_rate": "9.48"}]},
            ]
        }
        self.assertEqual(mu._frame_rate_v1(session), 9.48)

    def test_v1_none_when_absent(self):
        """v1 returns None when no fov carries a frame_rate."""
        self.assertIsNone(mu._frame_rate_v1({"data_streams": [{}]}))

    def test_v2_imaging_config_sampling_strategy(self):
        """v2 reads ImagingConfig.sampling_strategy.frame_rate."""
        acq = {
            "data_streams": [
                {
                    "configurations": [
                        {"object_type": "other"},
                        {
                            "object_type": mu._IMAGING_CONFIG_TYPE,
                            "sampling_strategy": {"frame_rate": 30.0},
                        },
                    ]
                }
            ]
        }
        self.assertEqual(mu._frame_rate_v2(acq), 30.0)

    def test_v2_none_when_absent(self):
        """v2 returns None with no ImagingConfig sampling strategy."""
        self.assertIsNone(mu._frame_rate_v2({"data_streams": []}))

    def test_platform_fallback(self):
        """platform.json frame rate is read from imaging_plane_groups."""
        platform = {
            "imaging_plane_groups": [{"acquisition_framerate_Hz": 6.2}]
        }
        self.assertEqual(mu._frame_rate_platform(platform), 6.2)
        self.assertIsNone(mu._frame_rate_platform({}))


class TestRequire(unittest.TestCase):
    """Unit tests for the generic fail-fast getter."""

    def test_returns_value(self):
        """_require returns the value when present."""
        self.assertEqual(mu._require({"k": 1}, "k", Path("f.json")), 1)

    def test_raises_named_error(self):
        """_require raises KeyError naming the field and file."""
        with self.assertRaises(KeyError) as ctx:
            mu._require({}, "instrument_id", Path("acquisition.json"))
        self.assertIn("instrument_id", str(ctx.exception))
        self.assertIn("acquisition.json", str(ctx.exception))


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
                                "object_type": mu._IMAGING_CONFIG_TYPE,
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


if __name__ == "__main__":
    unittest.main()

"""Tests for MotionCorrectionSettings."""

import unittest
from pathlib import Path

from aind_ophys_motion_correction_library.settings import (
    MotionCorrectionSettings,
)


class TestMotionCorrectionSettings(unittest.TestCase):
    """Settings defaults, typing, and env parsing."""

    def _settings(self, **kwargs) -> MotionCorrectionSettings:
        """Build settings without reading process CLI args."""
        return MotionCorrectionSettings(_cli_parse_args=[], **kwargs)

    def test_defaults(self):
        """Core defaults match the capsule's expectations."""
        s = self._settings()
        self.assertEqual(s.input_dir, Path("/data/"))
        self.assertEqual(s.output_dir, Path("/results/"))
        self.assertEqual(s.data_type, "h5")
        self.assertEqual(s.block_size, [128, 128])
        self.assertFalse(s.debug)

    def test_do_registration_is_bool(self):
        """do_registration is a real bool, not the legacy string."""
        s = self._settings()
        self.assertIsInstance(s.do_registration, bool)
        self.assertTrue(s.do_registration)

    def test_frame_rate_fallback_field(self):
        """frame_rate CLI fallback defaults to None and accepts floats."""
        self.assertIsNone(self._settings().frame_rate)
        self.assertEqual(self._settings(frame_rate=11.0).frame_rate, 11.0)

    def test_overrides(self):
        """Explicit values override defaults."""
        s = self._settings(data_type="bruker", nonrigid=False)
        self.assertEqual(s.data_type, "bruker")
        self.assertFalse(s.nonrigid)


if __name__ == "__main__":
    unittest.main()

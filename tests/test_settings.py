"""Tests for MotionCorrectionSettings."""

import unittest
from pathlib import Path

from aind_ophys_motion_correction_library.settings import (
    MotionCorrectionSettings,
)


class TestMotionCorrectionSettings(unittest.TestCase):
    """Settings defaults, typing, and env parsing."""

    def _settings(
        self, **kwargs: object
    ) -> MotionCorrectionSettings:
        """Build settings without reading process CLI args."""
        return MotionCorrectionSettings(_cli_parse_args=[], **kwargs)

    def test_defaults(self) -> None:
        """Core defaults match the capsule's expectations."""
        s = self._settings()
        self.assertEqual(s.input_dir, Path("/data/"))
        self.assertEqual(s.output_dir, Path("/results/"))
        self.assertEqual(s.data_type, "h5")
        self.assertEqual(s.block_size, [128, 128])
        self.assertFalse(s.debug)

    def test_do_registration_is_bool(self) -> None:
        """do_registration is a real bool, not the legacy string."""
        s = self._settings()
        self.assertIsInstance(s.do_registration, bool)
        self.assertTrue(s.do_registration)

    def test_frame_rate_fallback_field(self) -> None:
        """frame_rate CLI fallback defaults to None and accepts floats."""
        self.assertIsNone(self._settings().frame_rate)
        self.assertEqual(self._settings(frame_rate=11.0).frame_rate, 11.0)

    def test_block_size_cli_string_coerces_to_ints(self) -> None:
        """A CLI '[128, 128]' block_size yields a list of ints."""
        s = MotionCorrectionSettings(
            _cli_parse_args=["--block_size=[100, 120]"]
        )
        self.assertEqual(s.block_size, [100, 120])
        self.assertTrue(all(isinstance(x, int) for x in s.block_size))

    def test_block_size_json_string_coerces_to_ints(self) -> None:
        """A raw JSON-string block_size (env/kwarg) coerces to ints."""
        s = self._settings(block_size="[64, 96]")
        self.assertEqual(s.block_size, [64, 96])
        self.assertTrue(all(isinstance(x, int) for x in s.block_size))

    def test_verify_flag(self) -> None:
        """verify defaults False and is settable."""
        self.assertFalse(self._settings().verify)
        self.assertTrue(self._settings(verify=True).verify)

    def test_overrides(self) -> None:
        """Explicit values override defaults."""
        s = self._settings(data_type="bruker", nonrigid=False)
        self.assertEqual(s.data_type, "bruker")
        self.assertFalse(s.nonrigid)


if __name__ == "__main__":
    unittest.main()

"""Tests for the provenance path helper.

The symlink case is the one that matters: inside the pipeline ``/results`` and
``/data`` are symlinks into the task work directory, and comparing a resolved
root against an unresolved path silently leaked absolute container paths such
as ``/results/VISp_0/motion_correction`` into ``processing.json``.
"""

import os
import tempfile
import unittest
from pathlib import Path

from aind_ophys_motion_correction_library.utils.paths import relative_to_root


class TestRelativeToRoot(unittest.TestCase):
    """relative_to_root resolves both sides before comparing."""

    def test_plain_relative(self):
        """A path under the root becomes relative."""
        root = Path(tempfile.mkdtemp())
        target = root / "VISp_0" / "motion_correction"
        target.mkdir(parents=True)
        self.assertEqual(
            relative_to_root(root, target), "VISp_0/motion_correction"
        )

    def test_symlinked_root(self):
        """An unresolved symlink path still yields a relative result."""
        base = Path(tempfile.mkdtemp())
        real = base / "capsule" / "results"
        real.mkdir(parents=True)
        link = base / "results_link"
        os.symlink(real, link)

        target = real / "VISp_0" / "motion_correction"
        target.mkdir(parents=True)

        # Root resolved, path addressed through the symlink -- the shape that
        # used to fail and emit an absolute path.
        unresolved = link / "VISp_0" / "motion_correction"
        self.assertEqual(
            relative_to_root(real, unresolved), "VISp_0/motion_correction"
        )
        # And the reverse: symlinked root, resolved path.
        self.assertEqual(
            relative_to_root(link, target), "VISp_0/motion_correction"
        )

    def test_outside_root_falls_back_to_absolute(self):
        """A path genuinely outside the root keeps its absolute form."""
        root = Path(tempfile.mkdtemp())
        other = Path(tempfile.mkdtemp()) / "elsewhere.h5"
        self.assertEqual(relative_to_root(root, other), str(other))

    def test_nonexistent_paths_do_not_raise(self):
        """Resolution of not-yet-created paths still works."""
        root = Path(tempfile.mkdtemp())
        target = root / "plane_0" / "motion_correction"
        self.assertEqual(
            relative_to_root(root, target), "plane_0/motion_correction"
        )

    def test_input_file_relative(self):
        """The data-root case used for Code.input_data."""
        base = Path(tempfile.mkdtemp())
        real = base / "capsule" / "data"
        real.mkdir(parents=True)
        link = base / "data_link"
        os.symlink(real, link)
        movie = real / "VISp_0" / "VISp_0.h5"
        movie.parent.mkdir(parents=True)
        movie.touch()
        self.assertEqual(
            relative_to_root(real, link / "VISp_0" / "VISp_0.h5"),
            "VISp_0/VISp_0.h5",
        )


if __name__ == "__main__":
    unittest.main()

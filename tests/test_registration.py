"""Tests for the metadata-driven helpers in the registration module.

Covers the per-plane output directories motion correction creates (every
downstream stage keys on them), the v2 frame-rate read off
``acquisition.json``, the debug-truncation logging, and single-plane input
discovery, where the movie is named after the plane id rather than the
session. The ``run()`` orchestration that calls these helpers is tested in
``test_job``.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from aind_pophys_metadata import CoreMetadata

from aind_ophys_motion_correction_library import registration as reg
from aind_ophys_motion_correction_library.utils import (
    metadata_utils as mu,
)

RESOURCES = Path(__file__).parent / "resources"
V2_MULTIPLANE = RESOURCES / "acquisition_multiplane_v2.json"


def _v2_multiplane_dir() -> Path:
    """Stage the real v2 multiplane acquisition in a temp directory.

    Returns
    -------
    Path
        Directory containing ``acquisition.json``.
    """
    directory = Path(tempfile.mkdtemp())
    shutil.copy(V2_MULTIPLANE, directory / "acquisition.json")
    return directory


def _staged_bergamo_directory() -> Path:
    """A directory shaped like the real Bergamo converter's output.

    The converter writes its stitched movie to ``<plane_id>/<plane_id>.h5``
    (e.g. ``plane_0/plane_0.h5``), named after the plane id -- not the
    session, even though the session name is a distinct, plausible-looking
    value of its own.

    Returns
    -------
    Path
        Directory containing ``acquisition.json``, ``data_description.json``
        and the plane-id-named movie.
    """
    directory = Path(tempfile.mkdtemp())
    shutil.copy(
        RESOURCES / "acquisition_bergamo_v2.json",
        directory / "acquisition.json",
    )
    (directory / "data_description.json").write_text(
        json.dumps(
            {"name": "single-plane-ophys_999999_2025-01-01_00-00-00"}
        )
    )
    plane_dir = directory / "plane_0"
    plane_dir.mkdir()
    (plane_dir / "plane_0.h5").touch()
    return directory


class TestPlaneOutputDirectories(unittest.TestCase):
    """Motion correction names the directories every later step reads."""

    def test_multiplane_yields_eight_distinct_directories(self) -> None:
        """Each of the eight planes gets its own output directory."""
        metadata = mu.load_motion_correction_metadata(_v2_multiplane_dir())
        root = Path(tempfile.mkdtemp())
        directories = {
            reg.make_output_directory(
                root, mu.resolve_plane_id(metadata.fov_ids, directory_id=plane)
            )
            for plane in metadata.fov_ids
        }
        self.assertEqual(len(directories), 8)
        self.assertEqual(
            {d.parent.name for d in directories}, set(metadata.fov_ids)
        )
        for directory in directories:
            self.assertTrue(directory.is_dir())

    def test_session_name_would_have_collided(self) -> None:
        """The session name is one value for eight planes, so it cannot be
        the plane id."""
        directory = _v2_multiplane_dir()
        (directory / "data_description.json").write_text(
            json.dumps({"name": "multiplane_session"})
        )
        metadata = mu.load_motion_correction_metadata(directory)
        self.assertEqual(len(metadata.fov_ids), 8)
        self.assertEqual(len({metadata.dataset_name}), 1)


class TestFindSingleplaneInput(unittest.TestCase):
    """Bergamo input is found by the converter's plane-id-named layout."""

    def test_dataset_name_selector_would_not_have_found_it(self) -> None:
        """A session-name selector finds nothing under the converter's
        plane-id-named layout."""
        directory = _staged_bergamo_directory()
        with self.assertRaises(FileNotFoundError):
            reg.find_singleplane_input(
                directory,
                Path(tempfile.mkdtemp()),
                "plane_0",
                "single-plane-ophys_999999_2025-01-01_00-00-00",
            )


class TestMultiplaneFrameRate(unittest.TestCase):
    """A v2 acquisition no longer needs a v1 session.json."""

    def test_frame_rate_from_v2_acquisition(self) -> None:
        """The rate comes off the v2 imaging config."""
        directory = _v2_multiplane_dir()
        core = CoreMetadata.load(directory)
        self.assertAlmostEqual(
            reg.resolve_multiplane_frame_rate(directory, core), 10.63
        )

    def test_cli_override_when_metadata_silent(self) -> None:
        """An operator override still applies on the multiplane path."""
        directory = Path(tempfile.mkdtemp())
        (directory / "acquisition.json").write_text(
            json.dumps({"instrument_id": "MESO.2"})
        )
        core = CoreMetadata.load(directory)
        self.assertEqual(
            reg.resolve_multiplane_frame_rate(directory, core, 12.5), 12.5
        )

    def test_raises_named_error_without_any_source(self) -> None:
        """No rate anywhere raises ValueError, not a bare string."""
        directory = Path(tempfile.mkdtemp())
        (directory / "acquisition.json").write_text(
            json.dumps({"instrument_id": "MESO.2"})
        )
        core = CoreMetadata.load(directory)
        with self.assertRaises(ValueError):
            reg.resolve_multiplane_frame_rate(directory, core)


class TestDebugTruncation(unittest.TestCase):
    """The effective frame count is logged, not the requested one."""

    def _movie(self, n_frames: int) -> Path:
        """Write a tiny HDF5 movie of ``n_frames`` frames.

        Parameters
        ----------
        n_frames : int
            Number of frames to write.

        Returns
        -------
        Path
            Path to the movie.
        """
        path = Path(tempfile.mkdtemp()) / "VISp_0.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "data", data=np.zeros((n_frames, 2, 2), dtype="uint16")
            )
        return path

    def test_warns_when_movie_is_already_shorter(self) -> None:
        """An upstream debug cap is called out instead of passing silently."""
        movie = self._movie(100)
        scratch = Path(tempfile.mkdtemp())
        with self.assertLogs(reg.logger, level="WARNING") as logs:
            out = reg.truncate_for_debug(movie, "VISp_0", 10.71, scratch)
        self.assertIn("upstream debug cap", "".join(logs.output))
        self.assertIn("100 frames", "".join(logs.output))
        with h5py.File(out, "r") as f:
            self.assertEqual(f["data"].shape[0], 100)

    def test_logs_effective_count_when_truncation_applies(self) -> None:
        """A genuine truncation logs the count it actually wrote."""
        movie = self._movie(50)
        scratch = Path(tempfile.mkdtemp())
        with self.assertLogs(reg.logger, level="INFO") as logs:
            out = reg.truncate_for_debug(movie, "VISp_1", 0.1, scratch)
        self.assertIn("36 frames", "".join(logs.output))
        self.assertEqual(out.name, "VISp_1.h5")
        with h5py.File(out, "r") as f:
            self.assertEqual(f["data"].shape[0], 36)


class TestLoadInitialFrames(unittest.TestCase):
    """HDF5 input reads only the sampled frames, matching a full read."""

    def setUp(self) -> None:
        """Write a movie whose frame ``i`` is filled with ``i``."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.movie = Path(self._tmp.name) / "movie.h5"
        frames = np.arange(20, dtype="int16")[:, None, None] * np.ones(
            (1, 3, 3), dtype="int16"
        )
        with h5py.File(self.movie, "w") as f:
            f.create_dataset("data", data=frames)

    def _values(self, path: Path, **kwargs) -> list:
        """Return the per-frame values ``load_initial_frames`` selects.

        Parameters
        ----------
        path : Path
            HDF5 movie.
        **kwargs
            Passed to :func:`registration.load_initial_frames`.

        Returns
        -------
        list of int
            The value of each selected frame.
        """
        frames = reg.load_initial_frames(str(path), "data", **kwargs)
        return [int(frame[0, 0]) for frame in frames]

    def test_samples_evenly_within_the_trim(self) -> None:
        """The same frames as sampling a fully read, trimmed movie."""
        self.assertEqual(
            self._values(
                self.movie, n_frames=4, trim_frames_start=2, trim_frames_end=2
            ),
            [2, 6, 10, 14],
        )

    def test_more_frames_than_the_movie_returns_every_frame(self) -> None:
        """Requesting more frames than exist returns them all."""
        self.assertEqual(
            self._values(self.movie, n_frames=50), list(range(20))
        )

    def test_reads_a_virtual_reference(self) -> None:
        """A virtual dataset over the movie samples like the movie."""
        reference = Path(self._tmp.name) / "reference.h5"
        with h5py.File(self.movie, "r") as f:
            source = h5py.VirtualSource(f["data"])
            layout = h5py.VirtualLayout(
                shape=(10, 3, 3), dtype=f["data"].dtype
            )
            layout[0:10] = source[5:15]
        with h5py.File(reference, "w") as ref:
            ref.create_virtual_dataset("data", layout)
        self.assertEqual(
            self._values(reference, n_frames=5), [5, 7, 9, 11, 13]
        )


class TestGenerateSinglePlaneReference(unittest.TestCase):
    """The reference movie holds exactly the reference epochs' frames."""

    def _movie(self, directory: Path, epochs: dict) -> Path:
        """Write a movie whose frame ``i`` is filled with the value ``i``.

        Parameters
        ----------
        directory : Path
            Where to write the movie.
        epochs : dict
            ``epoch_locations``: stem to ``[start, stop]`` frame range.

        Returns
        -------
        Path
            The movie.
        """
        movie = directory / "plane_0.h5"
        frames = np.arange(12, dtype="int16")[:, None, None] * np.ones(
            (1, 4, 4), dtype="int16"
        )
        with h5py.File(movie, "w") as f:
            f.create_dataset("data", data=frames)
            f.create_dataset("epoch_locations", data=[json.dumps(epochs)])
        return movie

    def _reference_frames(self, document: dict, epochs: dict) -> list:
        """Build the reference movie and return its per-frame values.

        Parameters
        ----------
        document : dict
            The ``metadata.json`` content.
        epochs : dict
            ``epoch_locations`` of the movie.

        Returns
        -------
        list of int
            The value of each reference frame.
        """
        directory = Path(tempfile.mkdtemp())
        (directory / "metadata.json").write_text(json.dumps(document))
        core = CoreMetadata.load(directory)
        movie = self._movie(directory, epochs)
        reference = reg.generate_single_plane_reference(
            movie, core, directory
        )
        with h5py.File(reference, "r") as f:
            return [int(frame[0, 0]) for frame in f["data"][()]]

    def test_concatenates_reference_epochs_in_document_order(self) -> None:
        """Photostim frames are left out; the rest are joined in order."""
        epochs = {"a": [0, 4], "b": [4, 8], "c": [8, 12]}
        document = {
            "epochs": [
                {"stimulus_name": "a", "tiff_stem": "a"},
                {"stimulus_name": "b", "tiff_stem": "b", "photostim": True},
                {"stimulus_name": "c", "tiff_stem": "c"},
            ]
        }
        self.assertEqual(
            self._reference_frames(document, epochs),
            [0, 1, 2, 3, 8, 9, 10, 11],
        )

    def test_single_reference_epoch(self) -> None:
        """One reference epoch gives exactly its frame range."""
        epochs = {"a": [0, 4], "b": [4, 8]}
        document = {
            "epochs": [
                {"stimulus_name": "a", "tiff_stem": "a"},
                {"stimulus_name": "b", "tiff_stem": "b", "reference": True},
            ]
        }
        self.assertEqual(
            self._reference_frames(document, epochs), [4, 5, 6, 7]
        )


if __name__ == "__main__":
    unittest.main()

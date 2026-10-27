"""Tests for the job entrypoint: plane resolution and the output writers.

Plane resolution turns an input directory into this task's movie and its
canonical plane id. The writers consume the type Suite2P actually hands back
- a numpy movie, numpy offset/correlation arrays, per-block nonrigid arrays
and a numpy ops dict - at realistic shapes, so a list-shaped stand-in would
exercise different code than a real run does.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import suite2p
from aind_pophys_metadata import CoreMetadata

from aind_ophys_motion_correction_library import job
from aind_ophys_motion_correction_library import registration as reg
from aind_ophys_motion_correction_library.utils import (
    metadata_utils as mu,
)

# Small but real: a movie shape Suite2P would produce, kept short so the
# writers run in milliseconds.
N_FRAMES = 50
FRAME_HEIGHT = 512
FRAME_WIDTH = 512
N_BLOCKS = 9

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
    (e.g. ``plane_0/plane_0.h5``), named after the plane id.

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
            {"name": "single-plane-ophys_772414_2025-04-11_17-41-16"}
        )
    )
    plane_dir = directory / "plane_0"
    plane_dir.mkdir()
    (plane_dir / "plane_0.h5").touch()
    return directory


def _ops(n_frames: int = N_FRAMES, nonrigid: bool = False) -> dict:
    """Return a Suite2P ops dict with numpy registration outputs.

    Parameters
    ----------
    n_frames : int, optional
        Number of registered frames.
    nonrigid : bool, optional
        Whether to include the per-block nonrigid arrays.

    Returns
    -------
    dict
        The ops dict, as loaded from ``ops.npy``.
    """
    rng = np.random.default_rng(0)
    ops = {
        "Ly": FRAME_HEIGHT,
        "Lx": FRAME_WIDTH,
        "nframes": n_frames,
        "xoff": rng.integers(-5, 6, n_frames).astype("int32"),
        "yoff": rng.integers(-5, 6, n_frames).astype("int32"),
        "corrXY": rng.random(n_frames).astype("float32"),
    }
    if nonrigid:
        ops["xoff1"] = rng.normal(size=(n_frames, N_BLOCKS)).astype("float32")
        ops["yoff1"] = rng.normal(size=(n_frames, N_BLOCKS)).astype("float32")
        ops["corrXY1"] = rng.random((n_frames, N_BLOCKS)).astype("float32")
    return ops


def _reg_out(nonrigid: bool = False, n_frames: int = N_FRAMES) -> dict:
    """Return a registration result shaped like ``_load_ops_and_clip``.

    Parameters
    ----------
    nonrigid : bool, optional
        Whether the ops carry the per-block nonrigid arrays.
    n_frames : int, optional
        Number of registered frames.

    Returns
    -------
    dict
        Registered movie, ops and clipped-offset arrays.
    """
    ops = _ops(n_frames=n_frames, nonrigid=nonrigid)
    is_valid = np.ones(n_frames, dtype="bool")
    is_valid[:2] = False
    is_valid[-3:] = False
    return {
        "data": np.zeros(
            (n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype="uint16"
        ),
        "ops": ops,
        "delta_x": ops["xoff"].copy(),
        "delta_y": ops["yoff"].copy(),
        "clipped_indices": [np.int64(4), np.int64(7)],
        "is_valid": is_valid,
    }


def _args(output_dir: Path) -> dict:
    """Return the resolved run arguments the writers read.

    Parameters
    ----------
    output_dir : Path
        Directory the outputs are written into.

    Returns
    -------
    dict
        Trim counts, frame rate and the per-output file paths.
    """
    args = {
        "trim_frames_start": 2,
        "trim_frames_end": 3,
        "frame_rate_hz": 10.71,
        "refImg": np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype="int16"),
    }
    job._build_output_paths(args, output_dir, "VISp_0.h5")
    return args


def _suite2p_args(nonrigid: bool = False) -> dict:
    """Return the Suite2P ops motion correction submits.

    Parameters
    ----------
    nonrigid : bool, optional
        Value of the ``nonrigid`` flag.

    Returns
    -------
    dict
        A real ``suite2p.default_ops()`` dict with this run's overrides.
    """
    suite2p_args = suite2p.default_ops()
    suite2p_args["h5py"] = "/data/VISp_0/VISp_0.h5"
    suite2p_args["h5py_key"] = "data"
    suite2p_args["nonrigid"] = nonrigid
    suite2p_args["refImg"] = np.zeros(
        (FRAME_HEIGHT, FRAME_WIDTH), dtype="int16"
    )
    return suite2p_args


class TestJobPlaneResolution(unittest.TestCase):
    """run() resolves this task's movie and its canonical plane id.

    The plane id must come from the movie's own layout, not from the
    session-level ``dataset_name`` (identical across every plane), because it
    names this task's output directory and the aggregator ``DataProcess``.
    """

    def test_resolves_the_plane_id_named_movie(self) -> None:
        """job._resolve_plane_input finds plane_0/plane_0.h5."""
        directory = _staged_bergamo_directory()
        metadata = mu.load_motion_correction_metadata(directory)
        core = CoreMetadata.load(directory)
        settings = SimpleNamespace(
            input_dir=directory,
            output_dir=Path(tempfile.mkdtemp()),
            data_type="h5",
            frame_rate=None,
        )
        ctx = job._resolve_plane_input(settings, metadata, core)
        self.assertEqual(ctx["unique_id"], "plane_0")
        self.assertEqual(
            ctx["input_file"], directory / "plane_0" / "plane_0.h5"
        )

    def test_unique_id_comes_from_the_plane_directory(self) -> None:
        """job._resolve_plane_input keys on the V*-named plane directory."""
        directory = _v2_multiplane_dir()
        (directory / "data_description.json").write_text(
            json.dumps({"name": "multiplane_session"})
        )
        plane_dir = directory / "VISp_0"
        plane_dir.mkdir()
        movie = plane_dir / "VISp_0.h5"
        movie.touch()
        metadata = mu.load_motion_correction_metadata(directory)
        core = CoreMetadata.load(directory)
        settings = SimpleNamespace(
            input_dir=directory,
            output_dir=Path(tempfile.mkdtemp()),
            data_type="h5",
            frame_rate=None,
        )
        ctx = job._resolve_plane_input(settings, metadata, core)
        self.assertEqual(ctx["unique_id"], "VISp_0")
        self.assertNotEqual(ctx["unique_id"], metadata.dataset_name)
        self.assertEqual(ctx["input_file"], movie)


class TestRegistrationOutputParameters(unittest.TestCase):
    """Scalar summary statistics describing the registered movie."""

    def test_numpy_result_is_supported(self) -> None:
        """The offsets and validity mask arrive as numpy arrays."""
        output_dir = Path(tempfile.mkdtemp())
        summary = job._registration_output_parameters(
            _args(output_dir), _reg_out()
        )
        self.assertEqual(summary["frame_count"], N_FRAMES)
        self.assertEqual(summary["valid_frame_count"], N_FRAMES - 5)
        self.assertEqual(summary["frames_trimmed_start"], 2)
        self.assertEqual(summary["frames_trimmed_end"], 3)
        self.assertEqual(summary["outlier_frames_clipped"], 2)
        for key in (
            "max_abs_shift_x_pixels",
            "mean_abs_shift_y_pixels",
        ):
            self.assertIsInstance(summary[key], float)

    def test_zero_shift_movie_reports_shift_statistics(self) -> None:
        """An unshifted movie still reports zeros rather than omitting."""
        output_dir = Path(tempfile.mkdtemp())
        reg_out = _reg_out()
        reg_out["delta_x"] = np.zeros(N_FRAMES, dtype="int32")
        reg_out["delta_y"] = np.zeros(N_FRAMES, dtype="int32")
        summary = job._registration_output_parameters(
            _args(output_dir), reg_out
        )
        self.assertEqual(summary["max_abs_shift_x_pixels"], 0.0)
        self.assertEqual(summary["mean_abs_shift_y_pixels"], 0.0)

    def test_empty_shift_arrays_omit_shift_statistics(self) -> None:
        """A movie with no offsets reports counts and nothing invented."""
        output_dir = Path(tempfile.mkdtemp())
        reg_out = _reg_out(n_frames=0)
        summary = job._registration_output_parameters(
            _args(output_dir), reg_out
        )
        self.assertEqual(summary["frame_count"], 0)
        self.assertNotIn("max_abs_shift_x_pixels", summary)


class TestWriteMotionCsv(unittest.TestCase):
    """The motion-transform CSV is what every downstream reader parses."""

    def test_rigid_csv_holds_one_row_per_frame(self) -> None:
        """Rigid runs write the clipped offsets alongside the raw ones."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        reg_out = _reg_out()
        job._write_motion_csv(_suite2p_args(), args, reg_out)
        df = pd.read_csv(args["motion_diagnostics_output"])
        self.assertEqual(len(df), N_FRAMES)
        self.assertNotIn("nonrigid_x", df.columns)
        np.testing.assert_array_equal(df["x"].to_numpy(), reg_out["delta_x"])
        np.testing.assert_array_equal(
            df["is_valid"].to_numpy(), reg_out["is_valid"]
        )

    def test_nonrigid_csv_holds_per_block_arrays(self) -> None:
        """Nonrigid runs serialize one bracketed block list per frame."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        reg_out = _reg_out(nonrigid=True)
        job._write_motion_csv(_suite2p_args(nonrigid=True), args, reg_out)
        df = pd.read_csv(args["motion_diagnostics_output"])
        self.assertEqual(len(df), N_FRAMES)
        for column in ("nonrigid_x", "nonrigid_y", "nonrigid_corr"):
            first = df[column].iloc[0]
            self.assertTrue(first.startswith("["))
            self.assertEqual(len(first.split(",")), N_BLOCKS)

    def test_nonrigid_blocks_stay_on_one_line(self) -> None:
        """A wrapped array2string would break the CSV row structure."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        job._write_motion_csv(
            _suite2p_args(nonrigid=True), args, _reg_out(nonrigid=True)
        )
        text = Path(args["motion_diagnostics_output"]).read_text()
        self.assertEqual(len(text.splitlines()), N_FRAMES + 1)

    def test_nan_blocks_survive_the_csv_round_trip(self) -> None:
        """A non-finite block value parses back as NaN, not an error."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        reg_out = _reg_out(nonrigid=True)
        reg_out["ops"]["corrXY1"][0, 0] = np.nan
        job._write_motion_csv(_suite2p_args(nonrigid=True), args, reg_out)
        df = pd.read_csv(args["motion_diagnostics_output"])
        blocks = reg.parse_block_column(df["nonrigid_corr"])
        self.assertEqual(blocks.shape, (N_FRAMES, N_BLOCKS))
        self.assertTrue(np.isnan(blocks[0, 0]))

    def test_wide_block_grid_stays_on_one_row(self) -> None:
        """Hundreds of blocks per frame do not wrap onto a second line."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        reg_out = _reg_out(nonrigid=True)
        rng = np.random.default_rng(1)
        for key in ("xoff1", "yoff1", "corrXY1"):
            reg_out["ops"][key] = rng.normal(size=(N_FRAMES, 500)).astype(
                "float32"
            )
        job._write_motion_csv(_suite2p_args(nonrigid=True), args, reg_out)
        text = Path(args["motion_diagnostics_output"]).read_text()
        self.assertEqual(len(text.splitlines()), N_FRAMES + 1)
        df = pd.read_csv(args["motion_diagnostics_output"])
        self.assertEqual(
            reg.parse_block_column(df["nonrigid_x"]).shape, (N_FRAMES, 500)
        )


class TestWriteQualityControl(unittest.TestCase):
    """The per-plane quality_control.json written beside the figures."""

    def test_metrics_reference_the_figures_they_describe(self) -> None:
        """Each metric references the figure its description names."""
        results_root = Path(tempfile.mkdtemp())
        output_dir = results_root / "VISp_0" / "motion_correction"
        output_dir.mkdir(parents=True)
        job._write_quality_control(
            SimpleNamespace(output_dir=results_root),
            {"output_dir": output_dir, "unique_id": "VISp_0"},
            _args(output_dir),
        )
        document = json.loads(
            (output_dir / "quality_control.json").read_text()
        )
        references = {m["name"]: m["reference"] for m in document["metrics"]}
        self.assertEqual(
            references,
            {
                "VISp_0 Registration Summary": (
                    "VISp_0/motion_correction/VISp_0_registration_summary.png"
                ),
                "VISp_0 FOV Quality": (
                    "VISp_0/motion_correction/VISp_0_maximum_projection.png"
                ),
            },
        )
        self.assertEqual(
            {m["tags"]["evaluation"] for m in document["metrics"]},
            {"Motion Correction"},
        )


class TestWriteRegisteredH5(unittest.TestCase):
    """The registered movie, its metadata and the registration metrics."""

    def test_movie_metadata_and_metrics_are_written(self) -> None:
        """Data, reference image and reg_metrics all land in the HDF5."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        reg_out = _reg_out()
        reg_out["ops"]["regDX"] = np.zeros((10, 3), dtype="float32")
        reg_out["ops"]["regPC"] = np.zeros(
            (2, 10, FRAME_HEIGHT, FRAME_WIDTH), dtype="float32"
        )
        reg_out["ops"]["tPC"] = np.zeros((N_FRAMES, 10), dtype="float32")
        job._write_registered_h5(_suite2p_args(), args, reg_out)
        with h5py.File(args["motion_corrected_output"], "r") as f:
            self.assertEqual(
                f["data"].shape, (N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
            )
            self.assertEqual(f["ref_image"].shape, (FRAME_HEIGHT, FRAME_WIDTH))
            self.assertEqual(f["reg_metrics/regDX"].shape, (10, 3))
            self.assertEqual(f["reg_metrics/tPC"].shape, (N_FRAMES, 10))
            metadata = json.loads(f["metadata"][()].decode("utf-8"))
        self.assertNotIn("refImg", metadata)
        self.assertNotIn("refImg", metadata["suite2p_args"])
        self.assertEqual(metadata["trim_frames_start"], 2)

    def test_missing_metrics_write_empty_datasets(self) -> None:
        """Ops without registration metrics still produce the group."""
        output_dir = Path(tempfile.mkdtemp())
        args = _args(output_dir)
        job._write_registered_h5(_suite2p_args(), args, _reg_out())
        with h5py.File(args["motion_corrected_output"], "r") as f:
            self.assertEqual(f["reg_metrics/regDX"].size, 0)
            self.assertEqual(f["reg_metrics/regPC"].size, 0)


class TestStageAndTrim(unittest.TestCase):
    """S3 staging in _stage_and_trim."""

    def test_s3_input_is_staged_into_tmp_dir(self) -> None:
        """An S3-backed movie is copied into settings.tmp_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "mount" / "VISp_0.h5"
            source.parent.mkdir()
            source.write_bytes(b"movie")
            scratch = Path(tmp) / "scratch"
            scratch.mkdir()
            settings = SimpleNamespace(tmp_dir=scratch, data_type="h5")
            suite2p_args = {"h5py": str(source), "h5py_key": "data"}
            args = {"auto_remove_empty_frames": False}
            with (
                patch.object(reg, "is_S3", return_value=True),
                patch.object(reg, "check_and_warn_on_datatype"),
            ):
                job._stage_and_trim(settings, suite2p_args, args)
            staged = scratch / "VISp_0.h5"
            self.assertEqual(suite2p_args["h5py"], str(staged))
            self.assertEqual(staged.read_bytes(), b"movie")


if __name__ == "__main__":
    unittest.main()

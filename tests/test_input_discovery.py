"""Tests for input discovery, including inputs staged as directory symlinks.

Workflow engines such as Nextflow stage task inputs as symlinks into the
producing task's work directory, and ``Path.rglob`` does not descend into a
symlinked directory. Most tests build a real directory outside the search root
and leave only a directory symlink inside it, so a lookup passes only if it
follows the link. Decoy files pin which entries each lookup selects, and the
``pophys`` tests pin the preference the TIFF lookup applies.
"""

import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator

from aind_pophys_metadata import CoreMetadata

from aind_ophys_motion_correction_library import job
from aind_ophys_motion_correction_library import registration as reg
from aind_ophys_motion_correction_library.utils import metadata_utils as mu

RESOURCES = Path(__file__).parent / "resources"
V2_MULTIPLANE = RESOURCES / "acquisition_multiplane_v2.json"
V2_BERGAMO = RESOURCES / "acquisition_bergamo_v2.json"

# Name of the symlink the search root holds; it carries no plane-id text.
LINK_NAME = "staged"


@contextmanager
def _behind_symlink(files: Dict[str, str]) -> Iterator[Path]:
    """Yield a search root whose only entry is a symlink to a real directory.

    The real directory lives outside the root, where an upstream task's
    results sit under Nextflow, and holds ``files``.

    Parameters
    ----------
    files : dict of str to str
        Path relative to the real directory -> text content to write there.

    Yields
    ------
    Path
        The search root; its only entry is the symlink ``LINK_NAME``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        real = Path(tmp) / "upstream" / "results"
        real.mkdir(parents=True)
        for name, text in files.items():
            path = real / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        root = Path(tmp) / "task" / "data"
        root.mkdir(parents=True)
        (root / LINK_NAME).symlink_to(real, target_is_directory=True)
        yield root


def _resolve_tiff_input(root: Path) -> dict:
    """Resolve a tif-route acquisition's input through the job entrypoint.

    Parameters
    ----------
    root : Path
        Input directory; an ``acquisition.json`` is added to it.

    Returns
    -------
    dict
        The context ``job._resolve_plane_input`` builds.
    """
    shutil.copy(V2_BERGAMO, root / "acquisition.json")
    metadata = mu.load_motion_correction_metadata(root)
    core = CoreMetadata.load(root)
    with tempfile.TemporaryDirectory() as results:
        settings = SimpleNamespace(
            input_dir=root,
            output_dir=Path(results),
            data_type="tif",
            frame_rate=None,
        )
        return job._resolve_plane_input(settings, metadata, core)


class TestMetadataBehindSymlink(unittest.TestCase):
    """The shared metadata lookup follows a symlinked staging directory."""

    def test_core_and_description_are_found(self) -> None:
        """The core file and a sibling document are read through the link."""
        files = {
            "acquisition.json": V2_MULTIPLANE.read_text(),
            "data_description.json": json.dumps(
                {"name": "multiplane_session"}
            ),
        }
        with _behind_symlink(files) as root:
            metadata = mu.load_motion_correction_metadata(root)
        self.assertEqual(metadata.instrument_id, "MESO.2")
        self.assertEqual(metadata.dataset_name, "multiplane_session")
        self.assertEqual(len(metadata.fov_ids), 8)


class TestMultiplaneInputBehindSymlink(unittest.TestCase):
    """find_multiplane_input follows a symlinked staging directory."""

    def test_sidecar_names_the_movie(self) -> None:
        """A plane sidecar is read and the movie it names is located."""
        files = {
            "A_notes.txt": "decoy.h5",
            "plane_data/VISp_0.txt": "VISp_0.h5",
            "plane_data/VISp_0.h5": "",
        }
        with _behind_symlink(files) as root:
            h5_file, plane_id = reg.find_multiplane_input(root)
            expected = root / LINK_NAME / "plane_data" / "VISp_0.h5"
        self.assertEqual(h5_file, expected)
        self.assertEqual(plane_id, "VISp_0")

    def test_plane_directory_without_sidecar(self) -> None:
        """Without a sidecar, the first directory with a V in its name wins."""
        files = {
            "A_V.txt": "",
            "A_other/A_other.h5": "",
            "VISp_3/VISp_3.h5": "",
        }
        with _behind_symlink(files) as root:
            h5_file, plane_id = reg.find_multiplane_input(root)
            expected = root / LINK_NAME / "VISp_3" / "VISp_3.h5"
        self.assertEqual(h5_file, expected)
        self.assertEqual(plane_id, "VISp_3")


class TestSingleplaneInputBehindSymlink(unittest.TestCase):
    """find_singleplane_input follows a symlinked staging directory."""

    def test_movie_is_selected_by_path(self) -> None:
        """Only a movie whose path contains the selector is chosen."""
        files = {"other/other.h5": "", "plane_0/plane_0.h5": ""}
        with (
            _behind_symlink(files) as root,
            tempfile.TemporaryDirectory() as results,
        ):
            h5_file, output_dir = reg.find_singleplane_input(
                root, Path(results), "plane_0", "plane_0"
            )
            expected = root / LINK_NAME / "plane_0" / "plane_0.h5"
            expected_output = Path(results) / "plane_0" / "motion_correction"
        self.assertEqual(h5_file, expected)
        self.assertEqual(output_dir, expected_output)

    def test_first_sorted_match_wins_and_warns(self) -> None:
        """Several matching movies warn and the first sorted one is used."""
        files = {"plane_0_b/plane_0_b.h5": "", "plane_0/plane_0.h5": ""}
        with (
            _behind_symlink(files) as root,
            tempfile.TemporaryDirectory() as results,
            self.assertLogs(reg.logger, level="WARNING") as logs,
        ):
            h5_file, _ = reg.find_singleplane_input(
                root, Path(results), "plane_0", "plane_0"
            )
            expected = root / LINK_NAME / "plane_0" / "plane_0.h5"
        self.assertEqual(h5_file, expected)
        self.assertIn("2 movies contain", "".join(logs.output))


class TestTiffInputDiscovery(unittest.TestCase):
    """The tif route's pophys lookup: symlinks, preference and errors."""

    def test_pophys_is_found_behind_symlink(self) -> None:
        """A pophys directory below a symlinked session is located."""
        files = {"session/pophys/frame.tiff": ""}
        with _behind_symlink(files) as root:
            input_file = _resolve_tiff_input(root)["input_file"]
            expected = root / LINK_NAME / "session" / "pophys"
        self.assertEqual(input_file, expected)

    def test_nested_pophys_wins_over_direct_child(self) -> None:
        """A pophys inside a subdirectory beats one directly in the root."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pophys").mkdir()
            (root / "session" / "pophys").mkdir(parents=True)
            input_file = _resolve_tiff_input(root)["input_file"]
        self.assertEqual(input_file, root / "session" / "pophys")

    def test_direct_pophys_is_the_fallback(self) -> None:
        """A pophys directly in the root is used when none is nested."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pophys").mkdir()
            input_file = _resolve_tiff_input(root)["input_file"]
        self.assertEqual(input_file, root / "pophys")

    def test_missing_pophys_names_the_name_and_directory(self) -> None:
        """No pophys anywhere raises an error naming it and the root."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError) as caught:
                _resolve_tiff_input(root)
        self.assertIn("'pophys'", str(caught.exception))
        self.assertIn(str(root), str(caught.exception))


if __name__ == "__main__":
    unittest.main()

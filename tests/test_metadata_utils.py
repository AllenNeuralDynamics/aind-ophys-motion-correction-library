"""Tests for the metadata read layer and the code-identity builder.

Normalization of each schema version lives in the shared
``aind_pophys_metadata`` library (tested there); these tests cover the
motion-correction surface: ``load_motion_correction_metadata``, canonical
plane-id resolution, the Bergamo conditioning-epoch stem, and the ``Code``
block motion correction stamps onto every emitted document.
"""

import json
import os
import shutil
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from aind_pophys_metadata import CoreMetadata, io

from aind_ophys_motion_correction_library.utils import metadata_utils as mu

LIBRARY = "aind-ophys-motion-correction-library"

RESOURCES = Path(__file__).parent / "resources"
V2_MULTIPLANE = RESOURCES / "acquisition_multiplane_v2.json"
V2_BERGAMO = RESOURCES / "acquisition_bergamo_v2.json"

# Key an epoch's ``output_parameters`` carries its TIFF stem under.
TIFF_STEM_KEY = "tiff_stem"

# Stem of the first conditioning epoch in the real Bergamo fixtures.
BERGAMO_STEM = "neuron3"

# Every plane of the v2 fixture, in acquisition order.
V2_FOV_IDS = (
    "VISp_0",
    "VISp_1",
    "VISp_2",
    "VISp_3",
    "VISp_4",
    "VISp_5",
    "VISp_6",
    "VISp_7",
)


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write a JSON payload into directory/name.

    Parameters
    ----------
    directory : Path
        Destination directory.
    name : str
        File name.
    payload : dict
        JSON-serializable payload.
    """
    (directory / name).write_text(json.dumps(payload))


def _imaging_config_type() -> str:
    """Return the v2 ImagingConfig discriminator string.

    Returns
    -------
    str
        The discriminator value the installed schema serializes.
    """
    from aind_data_schema.components.configs import ImagingConfig

    return io.object_type_value(ImagingConfig)


def _v2_multiplane_dir() -> Path:
    """Stage the real v2 multiplane fixture in a temp directory.

    Returns
    -------
    Path
        Directory containing ``acquisition.json`` and a session-level
        ``data_description.json``.
    """
    directory = Path(tempfile.mkdtemp())
    shutil.copy(V2_MULTIPLANE, directory / "acquisition.json")
    _write(directory, "data_description.json", {"name": "multiplane_session"})
    return directory


class TestLoadMetadata(unittest.TestCase):
    """File-presence dispatch and independent id reads."""

    def test_v2_dispatch(self) -> None:
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

    def test_v1_dispatch(self) -> None:
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
        self.assertEqual(md.fov_ids, ("plane_0",))

    def test_subject_id_canonical_from_core(self) -> None:
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

    def test_platform_fallback_used(self) -> None:
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

    def test_cli_override_used_when_metadata_silent(self) -> None:
        """The operator override is the last frame-rate source."""
        d = Path(tempfile.mkdtemp())
        _write(d, "session.json", {"rig_id": "MESO.2"})
        md = mu.load_motion_correction_metadata(d, cli_override=7.5)
        self.assertEqual(md.frame_rate, 7.5)

    def test_frame_rate_absent_is_not_fatal_here(self) -> None:
        """A rate-less asset loads; the caller decides if that is fatal."""
        d = Path(tempfile.mkdtemp())
        _write(d, "session.json", {"rig_id": "MESO.2"})
        self.assertIsNone(mu.load_motion_correction_metadata(d).frame_rate)

    def test_raises_without_core_file(self) -> None:
        """Neither core file present -> FileNotFoundError."""
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            mu.load_motion_correction_metadata(d)

    def test_v2_multiplane_fixture_planes(self) -> None:
        """The real upgraded v2 acquisition yields eight canonical ids."""
        md = mu.load_motion_correction_metadata(_v2_multiplane_dir())
        self.assertEqual(md.fov_ids, V2_FOV_IDS)
        self.assertEqual(md.instrument_id, "MESO.2")
        self.assertAlmostEqual(md.frame_rate, 10.63)

    def test_core_is_not_reread(self) -> None:
        """A pre-loaded CoreMetadata is used instead of a second read."""
        d = _v2_multiplane_dir()
        core = CoreMetadata.load(d)
        md = mu.load_motion_correction_metadata(d, core=core)
        self.assertEqual(md.fov_ids, V2_FOV_IDS)


class TestResolvePlaneId(unittest.TestCase):
    """Canonical plane identity, and the degraded cases that must not raise."""

    def test_single_plane_uses_canonical_id(self) -> None:
        """One canonical plane wins over any caller fallback."""
        self.assertEqual(
            mu.resolve_plane_id(("plane_0",), fallback="session_name"),
            "plane_0",
        )

    def test_no_planes_falls_back_with_warning(self) -> None:
        """Metadata declaring no planes keeps the caller's id."""
        with self.assertLogs(mu.logger, level="WARNING"):
            resolved = mu.resolve_plane_id((), fallback="session_name")
        self.assertEqual(resolved, "session_name")

    def test_many_planes_without_directory_id_falls_back(self) -> None:
        """A single-plane route on multiplane metadata cannot guess."""
        with self.assertLogs(mu.logger, level="WARNING"):
            resolved = mu.resolve_plane_id(V2_FOV_IDS, fallback="plane_0")
        self.assertEqual(resolved, "plane_0")

    def test_directory_id_wins(self) -> None:
        """A directory-derived id names this task's outputs."""
        self.assertEqual(
            mu.resolve_plane_id(V2_FOV_IDS, directory_id="VISp_3"), "VISp_3"
        )

    def test_directory_id_without_canonical_ids(self) -> None:
        """No canonical ids at all still keeps the directory-derived one."""
        self.assertEqual(
            mu.resolve_plane_id((), directory_id="VISp_3"), "VISp_3"
        )


def _staged_core(core_name: str, core: dict) -> CoreMetadata:
    """Load a core document written to a temp directory.

    Parameters
    ----------
    core_name : str
        File name to write the core document as.
    core : dict
        The core document.

    Returns
    -------
    aind_pophys_metadata.CoreMetadata
        Metadata read back through the real load path.
    """
    directory = Path(tempfile.mkdtemp())
    _write(directory, core_name, core)
    return CoreMetadata.load(directory)


class TestBciEpochTiffStem(unittest.TestCase):
    """The Bergamo reference epoch reads from either stem route."""

    def test_stem_from_epoch_output_parameters(self) -> None:
        """A document carrying the stem on the epoch itself resolves."""
        core = _staged_core(
            "session.json",
            {
                "rig_id": "Bergamo.1",
                "stimulus_epochs": [
                    {"stimulus_name": "spontaneous activity"},
                    {
                        "stimulus_name": mu.BCI_STIMULUS_NAME,
                        "output_parameters": {
                            TIFF_STEM_KEY: BERGAMO_STEM
                        },
                    },
                ],
            },
        )
        self.assertEqual(mu.bci_epoch_tiff_stem(core), BERGAMO_STEM)

    def test_stem_from_data_stream_note(self) -> None:
        """A document carrying only the stream note resolves identically."""
        core = json.loads(V2_BERGAMO.read_text())
        core = _staged_core("acquisition.json", core)
        self.assertEqual(mu.bci_epoch_tiff_stem(core), BERGAMO_STEM)

    def test_both_routes_agree(self) -> None:
        """Adding the epoch-side stem does not change the result."""
        core = json.loads(V2_BERGAMO.read_text())
        for epoch in core["stimulus_epochs"]:
            epoch["output_parameters"] = {TIFF_STEM_KEY: BERGAMO_STEM}
        core = _staged_core("acquisition.json", core)
        self.assertEqual(mu.bci_epoch_tiff_stem(core), BERGAMO_STEM)

    def test_first_matching_epoch_in_document_order_wins(self) -> None:
        """Document order picks the epoch; it is not sorted by time."""
        core = json.loads(V2_BERGAMO.read_text())
        core["stimulus_epochs"] = list(reversed(core["stimulus_epochs"]))
        core = _staged_core("acquisition.json", core)
        self.assertEqual(mu.bci_epoch_tiff_stem(core), "neuron3_2")

    def test_stemless_epoch_raises_named_error(self) -> None:
        """Neither route present raises ValueError, not a bare KeyError."""
        core = json.loads(V2_BERGAMO.read_text())
        for stream in core["data_streams"]:
            stream["notes"] = None
        core = _staged_core("acquisition.json", core)
        with self.assertRaises(ValueError) as caught:
            mu.bci_epoch_tiff_stem(core)
        message = str(caught.exception)
        self.assertIn(mu.BCI_STIMULUS_NAME, message)
        self.assertIn("acquisition.json", message)

    def test_no_conditioning_epoch_raises(self) -> None:
        """An acquisition with no conditioning epoch degrades to the error."""
        core = CoreMetadata.load(_v2_multiplane_dir())
        with self.assertRaises(ValueError):
            mu.bci_epoch_tiff_stem(core)


class TestBuildCode(unittest.TestCase):
    """Code identity: a resolvable version and a matching url."""

    @patch.dict(os.environ, {"VERSION": "3.2.1"}, clear=True)
    def test_identity_is_the_installed_library(self) -> None:
        """name, version and url describe this library; VERSION is ignored."""
        code = mu.build_code({"a": 1})
        self.assertEqual(code.name, LIBRARY)
        self.assertEqual(code.version, version(LIBRARY))
        self.assertNotEqual(code.version, "3.2.1")
        self.assertEqual(
            code.url, f"https://github.com/AllenNeuralDynamics/{LIBRARY}"
        )

    def test_code_url_has_no_branch_ref(self) -> None:
        """The recorded url is a landing page, never a branch subpath."""
        self.assertNotIn("/tree/", mu.build_code({}).url)

    def test_input_data_optional(self) -> None:
        """input_data=None yields a Code without input assets."""
        self.assertIsNone(mu.build_code({}).input_data)

    def test_input_data_recorded(self) -> None:
        """Input assets are recorded when supplied."""
        code = mu.build_code({}, ["VISp_0/VISp_0.h5"])
        self.assertEqual(code.input_data[0].name, "VISp_0/VISp_0.h5")


if __name__ == "__main__":
    unittest.main()

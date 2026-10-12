"""Acquisition metadata reads for motion correction.

A thin, motion-correction-specific layer over the shared
``aind_pophys_metadata`` library. The shared library owns schema dispatch and
the output builders; this module only assembles the
``MotionCorrectionMetadata`` view from the shared ``CoreMetadata`` reads,
resolves this task's canonical plane id, and names the code artifact.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from aind_data_schema.components.identifiers import Code
from aind_pophys_metadata import CoreMetadata
from aind_pophys_metadata import processing as shared_processing

logger = logging.getLogger(__name__)

# Human-readable name of the motion-correction code artifact.
CODE_NAME = "aind-ophys-motion-correction"

# Distribution name of this backing library; the sole source of
# ``Code.version``.
LIBRARY_NAME = "aind-ophys-motion-correction-library"

# Repository this code lives in, recorded as ``Code.url``.
CODE_URL = (
    "https://github.com/AllenNeuralDynamics/"
    "aind-ophys-motion-correction-library"
)

# Plane id used when the acquisition metadata declares no planes at all.
DEFAULT_PLANE_ID = "plane_0"

# Bergamo single-plane acquisitions take their reference movie from the first
# conditioning epoch.
BCI_STIMULUS_NAME = "single neuron BCI conditioning"


@dataclass(frozen=True)
class MotionCorrectionMetadata:
    """Minimal metadata motion correction consumes.

    Attributes
    ----------
    frame_rate : float or None
        Acquisition frame rate in Hz; drives temporal detrending and preview
        downsampling. ``None`` when no source supplied one.
    instrument_id : str or None
        Instrument/rig identifier, used only to route Bergamo acquisitions.
    subject_id : str or None
        Subject identifier, used only for logging.
    dataset_name : str or None
        Session name from ``data_description.json``. Session-level, so it is
        never a plane id - it is identical for every plane of a multiplane
        acquisition.
    fov_ids : tuple of str
        Canonical plane/FOV ids for the whole acquisition, in plane order.
        Empty when the core file declares no planes.
    """

    frame_rate: Optional[float] = None
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None
    fov_ids: Tuple[str, ...] = ()


def load_motion_correction_metadata(
    input_dir: Path,
    *,
    core: Optional[CoreMetadata] = None,
    cli_override: Optional[float] = None,
) -> MotionCorrectionMetadata:
    """Read acquisition metadata from ``input_dir``.

    ``frame_rate`` falls back to ``platform.json`` and then to
    ``cli_override``; it is not required here, because the multiplane path has
    a further sync-file fallback of its own.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    core : aind_pophys_metadata.CoreMetadata, optional
        Pre-loaded core metadata, so a caller that already needs it does
        not read the documents twice.
    cli_override : float, optional
        Operator-supplied frame rate, used when the metadata provides none.

    Returns
    -------
    MotionCorrectionMetadata
        The minimal fields motion correction consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If the input is ambiguous or its core file is unreadable.
    """
    input_dir = Path(input_dir)
    core = core if core is not None else CoreMetadata.load(input_dir)
    return MotionCorrectionMetadata(
        frame_rate=core.get_frame_rate(
            cli_override=cli_override, required=False
        ),
        instrument_id=core.get_instrument_id(),
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
        fov_ids=core.get_fov_ids(),
    )


def resolve_plane_id(
    fov_ids: Sequence[str],
    directory_id: Optional[str] = None,
    fallback: str = DEFAULT_PLANE_ID,
) -> str:
    """Return the plane id this task's output directory is named after.

    The per-plane id names this task's output directory, which every
    downstream stage keys on, so it must be the canonical id from the
    acquisition metadata (see
    :meth:`aind_pophys_metadata.CoreMetadata.get_fov_ids`). When no canonical
    id can be chosen, falls back to the directory-derived id and warns rather
    than raising.

    Parameters
    ----------
    fov_ids : sequence of str
        Canonical plane ids for the whole acquisition.
    directory_id : str, optional
        Plane id recovered from the input directory layout, which wins
        outright. Pass it for a multiplane task, where only the input file
        says which plane this task holds; omit it for a single-plane task,
        where the canonical id is unambiguous.
    fallback : str, optional
        Id to use when ``directory_id`` is omitted and no single canonical id
        can be chosen.

    Returns
    -------
    str
        The plane id to name this task's outputs with.
    """
    if directory_id is not None:
        return directory_id
    ids = tuple(fov_ids or ())
    if len(ids) == 1:
        return ids[0]
    if not ids:
        logger.warning(
            "Acquisition metadata declares no planes; naming outputs "
            "after the directory-derived id %r.",
            fallback,
        )
    else:
        logger.warning(
            "Acquisition metadata declares %d planes %s but this task "
            "resolves a single plane; naming outputs after the "
            "directory-derived id %r.",
            len(ids),
            list(ids),
            fallback,
        )
    return fallback


def bci_epoch_tiff_stem(core: CoreMetadata) -> str:
    """Return the TIFF stem of the first BCI conditioning epoch.

    Epochs are read in document order, which is the order the acquisition
    recorded and is not necessarily chronological.

    Parameters
    ----------
    core : aind_pophys_metadata.CoreMetadata
        The located acquisition metadata document.

    Returns
    -------
    str
        The conditioning epoch's TIFF stem.

    Raises
    ------
    ValueError
        If no conditioning epoch carries a stem. The message names the
        stimulus and the core file, so an acquisition missing the stem is
        distinguishable from an unrelated one.
    """
    for epoch in core.get_epoch_records():
        if epoch.get("stimulus_name") != BCI_STIMULUS_NAME:
            continue
        if epoch.get("tiff_stem"):
            return epoch["tiff_stem"]
    raise ValueError(
        f"No {BCI_STIMULUS_NAME!r} stimulus epoch carrying a tiff stem was "
        f"found in {Path(core.core_path).name}; the Bergamo reference "
        "movie cannot be built."
    )


def build_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the motion-correction ``Code`` block.

    Parameters
    ----------
    parameters : dict
        Run configuration parameters (from ``MotionCorrectionSettings``).
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block: :data:`CODE_URL` for the repository, and
        the installed backing library for the version.
    """
    return shared_processing.build_code(
        CODE_NAME,
        library_name=LIBRARY_NAME,
        url=CODE_URL,
        parameters=parameters,
        input_data=input_data,
    )

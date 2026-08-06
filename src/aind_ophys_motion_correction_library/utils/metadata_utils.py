"""aind-data-schema v1/v2 metadata reads for motion correction.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, the ``Code`` builder
and the ``processing.json`` lifecycle live in the shared library; this module
only assembles the ``MotionCorrectionMetadata`` view, resolves this task's
canonical plane id, and names the motion-correction code artifact.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aind_data_schema.components.identifiers import Code
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing

logger = logging.getLogger(__name__)

# Human-readable name of the motion-correction code artifact.
CODE_NAME = "aind-ophys-motion-correction"

# Distribution name of this backing library. The shared builder derives both
# ``Code.url`` and ``Code.version`` from it, so the two always name one
# artifact: the released library.
LIBRARY_NAME = "aind-ophys-motion-correction-library"

# Plane id used when the acquisition metadata declares no planes at all.
DEFAULT_PLANE_ID = "plane_0"

# Bergamo single-plane acquisitions take their reference movie from the first
# conditioning epoch.
BCI_STIMULUS_NAME = "single neuron BCI conditioning"

# v1 carried the TIFF stem on ``stimulus_epochs[*].output_parameters``. The
# v1 -> v2 upgrade drops that field outright, but preserves the same value
# verbatim on the correspondingly-timed data stream's ``notes``.
TIFF_STEM_KEY = "tiff_stem"
TIFF_STEM_NOTE_PREFIX = f"{TIFF_STEM_KEY}:"


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
    common: Optional[io.CommonMetadata] = None,
    cli_override: Optional[float] = None,
) -> MotionCorrectionMetadata:
    """Read v1/v2 metadata for motion correction from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). ``frame_rate`` falls back to ``platform.json``
    and then to ``cli_override``; it is not required here, because the
    multiplane path has a further sync-file fallback of its own.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    common : aind_pophys_metadata.io.CommonMetadata, optional
        Pre-loaded core/sibling documents, so a caller that already needs the
        raw dicts does not read them twice.
    cli_override : float, optional
        Operator-supplied frame rate, used when the metadata provides none.

    Returns
    -------
    MotionCorrectionMetadata
        The minimal fields this capsule consumes.

    Raises
    ------
    FileNotFoundError
        If neither ``acquisition.json`` nor ``session.json`` is found.
    ValueError
        If both core files are present (ambiguous input).
    """
    input_dir = Path(input_dir)
    common = common if common is not None else io.load_common(input_dir)
    return MotionCorrectionMetadata(
        frame_rate=fields.resolve_frame_rate(
            common.version,
            common.core_raw,
            common.platform_raw,
            core_file=common.core_path,
            input_dir=input_dir,
            cli_override=cli_override,
            required=False,
        ),
        instrument_id=fields.get_instrument_id(
            common.version, common.core_raw
        ),
        subject_id=fields.get_subject_id(
            common.core_raw, common.subject_raw
        ),
        dataset_name=fields.get_dataset_name(common.data_description_raw),
        fov_ids=fields.get_fov_ids(common.version, common.core_raw),
    )


def resolve_plane_id(
    fov_ids: Sequence[str],
    scavenged: Optional[str] = None,
    fallback: str = DEFAULT_PLANE_ID,
) -> str:
    """Return the plane id this task's output directory is named after.

    Motion correction creates the per-plane directories every downstream
    capsule keys on, so its plane id must be the canonical one derived from
    the acquisition metadata (see
    :func:`aind_pophys_metadata.fields.get_fov_ids`). Metadata that declares
    no planes still has to run, so every degraded case falls back to the
    caller's directory-derived id and warns rather than raising.

    Parameters
    ----------
    fov_ids : sequence of str
        Canonical plane ids for the whole acquisition.
    scavenged : str, optional
        Plane id recovered from the input directory layout. Pass it for a
        multiplane task, where the canonical set names every plane and only
        the input file says which one this task holds; omit it for a
        single-plane task, where the canonical id is unambiguous.
    fallback : str, optional
        Id to use when ``scavenged`` is omitted and no single canonical id
        can be chosen.

    Returns
    -------
    str
        The plane id to name this task's outputs with.
    """
    ids = tuple(fov_ids or ())
    if scavenged is None:
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
    if scavenged in ids:
        return scavenged
    if not ids:
        logger.warning(
            "Acquisition metadata declares no planes; keeping the "
            "directory-derived plane id %r.",
            scavenged,
        )
    else:
        fields.validate_scavenged_ids([scavenged], ids, logger)
    return scavenged


def _v2_stems_by_start_time(core_raw: dict) -> Dict[Any, str]:
    """Index v2 data-stream TIFF stems by stream start time.

    Parameters
    ----------
    core_raw : dict
        Raw ``acquisition.json`` dict.

    Returns
    -------
    dict
        Mapping of ``stream_start_time`` to TIFF stem.
    """
    stems: Dict[Any, str] = {}
    for stream in core_raw.get("data_streams") or []:
        notes = stream.get("notes") or ""
        if notes.startswith(TIFF_STEM_NOTE_PREFIX):
            stems[stream.get("stream_start_time")] = notes[
                len(TIFF_STEM_NOTE_PREFIX):
            ]
    return stems


def bci_epoch_tiff_stem(core_raw: dict, version: str) -> str:
    """Return the TIFF stem of the first BCI conditioning epoch.

    v1 records the stem on the epoch itself. The v1 -> v2 upgrade drops
    ``stimulus_epochs[*].output_parameters`` entirely, so the v2 read
    recovers it from the data stream sharing the epoch's start time - the
    only place the value survives, verified against real upgrader output.
    Matching is by start time rather than by list position, because the
    upgrade is free to drop or reorder epochs.

    Parameters
    ----------
    core_raw : dict
        Raw ``session.json`` (v1) or ``acquisition.json`` (v2) dict.
    version : str
        Schema version string (see
        :func:`aind_pophys_metadata.io.detect_schema_version`).

    Returns
    -------
    str
        The conditioning epoch's TIFF stem.

    Raises
    ------
    ValueError
        If no conditioning epoch yields a stem. The message names the
        stimulus and the schema version, so a v2 document whose stream notes
        were lost is distinguishable from an unrelated acquisition.
    """
    is_v1 = version == io.SCHEMA_V1
    stems_by_time = {} if is_v1 else _v2_stems_by_start_time(core_raw)
    for epoch in core_raw.get("stimulus_epochs") or []:
        if epoch.get("stimulus_name") != BCI_STIMULUS_NAME:
            continue
        if is_v1:
            stem = (epoch.get("output_parameters") or {}).get(TIFF_STEM_KEY)
        else:
            stem = stems_by_time.get(epoch.get("stimulus_start_time"))
        if stem:
            return stem
    raise ValueError(
        f"No {BCI_STIMULUS_NAME!r} stimulus epoch yielding a "
        f"{TIFF_STEM_KEY!r} was found in the schema-{version} acquisition "
        "metadata; the Bergamo reference movie cannot be built."
    )


def build_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the motion-correction ``Code`` block.

    Parameters
    ----------
    parameters : dict
        Capsule configuration parameters (from ``MotionCorrectionSettings``).
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block. Url and version both come from this
        library's distribution, so the two never describe different
        artifacts.
    """
    return shared_processing.build_code(
        CODE_NAME,
        library_name=LIBRARY_NAME,
        parameters=parameters,
        input_data=input_data,
    )

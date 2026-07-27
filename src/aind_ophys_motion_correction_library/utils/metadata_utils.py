"""aind-data-schema v1/v2 metadata reads for motion correction.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, and the
``Processing`` builders live in the shared library; this module only assembles
the motion-correction ``MotionCorrectionMetadata`` view and its
motion-correction-specific ``DataProcess``.
"""

import os
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional

import pytz
from aind_data_schema.core.processing import DataProcess, Processing
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

CODE_URL = (
    "https://github.com/AllenNeuralDynamics/"
    "aind-ophys-motion-correction/tree/main/code"
)


@dataclass(frozen=True)
class MotionCorrectionMetadata:
    """Minimal metadata motion correction consumes.

    ``frame_rate`` drives temporal detrending and preview downsampling;
    the remaining fields are identifiers used only for routing, logging,
    and output naming.
    """

    frame_rate: Optional[float] = None
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None


def load_motion_correction_metadata(
    input_dir: Path,
) -> MotionCorrectionMetadata:
    """Read v1/v2 metadata for motion correction from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). ``frame_rate`` falls back to ``platform.json``
    when the core file lacks it; the caller applies any final CLI default.
    ``subject_id`` is read from the core file (canonical) and falls back to
    ``subject.json``; ``dataset_name`` comes from ``data_description.json``.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.

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
    core_path = io.find_acquisition_file(input_dir)
    version = io.detect_schema_version(core_path)
    core_raw = io.load_json(core_path)

    platform_path = io.find(input_dir, "platform.json")
    platform_raw = io.load_json(platform_path) if platform_path else None
    subject_path = io.find(input_dir, "subject.json")
    subject_raw = io.load_json(subject_path) if subject_path else None
    dd_path = io.find(input_dir, "data_description.json")
    dd_raw = io.load_json(dd_path) if dd_path else None

    return MotionCorrectionMetadata(
        frame_rate=fields.get_frame_rate(version, core_raw, platform_raw),
        instrument_id=fields.get_instrument_id(version, core_raw),
        subject_id=fields.get_subject_id(core_raw, subject_raw),
        dataset_name=fields.get_dataset_name(dd_raw),
    )


def build_data_process(
    parameters: dict,
    start_time: dt,
    end_time: dt,
    input_data: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    output_parameters: Optional[dict] = None,
) -> DataProcess:
    """Construct a v2 DataProcess for motion correction.

    Parameters
    ----------
    parameters : dict
        Capsule configuration parameters (from MotionCorrectionSettings).
    start_time : dt
        Processing start time.
    end_time : dt
        Processing end time.
    input_data : list of str, optional
        Input data asset name(s) for provenance.
    output_path : str, optional
        Relative path to the output directory for this process.
    output_parameters : dict, optional
        Actual parameters used and output metrics.

    Returns
    -------
    DataProcess
        The populated motion-correction data process.
    """
    code = shared_processing.build_code(
        url=CODE_URL,
        name="aind-ophys-motion-correction",
        version=os.getenv("VERSION", ""),
        parameters=parameters,
        input_data=input_data,
    )
    return shared_processing.build_data_process(
        process_type=ProcessName.VIDEO_MOTION_CORRECTION,
        code=code,
        start_time=start_time.astimezone(_SEATTLE_TZ),
        end_time=end_time.astimezone(_SEATTLE_TZ),
        name="Suite2P motion correction",
        output_path=output_path,
        output_parameters=output_parameters,
    )


def build_processing(data_processes: List[DataProcess]) -> Processing:
    """Wrap data processes into a full v2 Processing object."""
    return shared_processing.build_processing(data_processes)


def write_processing_json(processing: Processing, output_dir: Path) -> Path:
    """Serialize a Processing object to ``processing.json``.

    Parameters
    ----------
    processing : Processing
        The processing document to write.
    output_dir : Path
        Directory into which ``processing.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_processing.write_processing_json(processing, output_dir)

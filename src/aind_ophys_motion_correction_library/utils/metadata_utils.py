"""aind-data-schema v1/v2 metadata reads for motion correction.

Single point of contact for ``aind-data-schema`` / ``-models``. Reads raw
dicts (no Pydantic validation on the read path), requires only the fields
this capsule consumes, and fails fast with a clear error. Schema version is
dispatched by which core acquisition file is present: ``session.json`` -> v1,
``acquisition.json`` -> v2.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Any, List, Optional

import pytz
from aind_data_schema.components.configs import ImagingConfig
from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.processing import (
    DataProcess,
    Processing,
    ProcessStage,
)
from aind_data_schema_models.process_names import ProcessName

# The v2 ImagingConfig discriminator, read from the schema rather than
# hardcoded, so it tracks schema re-wordings.
_IMAGING_CONFIG_TYPE = ImagingConfig.model_fields["object_type"].default

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


def _require(blob: dict, key: str, file_path: Path) -> Any:
    """Return ``blob[key]`` or raise a KeyError naming the field + file."""
    if key not in blob:
        raise KeyError(f"Required field '{key}' missing from {file_path.name}")
    return blob[key]


def _load_json(path: Path) -> dict:
    """Parse a JSON file into a dict."""
    with open(path) as f:
        return json.load(f)


def _find(input_dir: Path, name: str) -> Optional[Path]:
    """Return the first match for ``name`` under ``input_dir``, or None."""
    return next(Path(input_dir).rglob(name), None)


def _frame_rate_v1(session: dict) -> Optional[float]:
    """Frame rate from a v1 session: first ophys_fov with a frame_rate."""
    for stream in session.get("data_streams") or []:
        fovs = stream.get("ophys_fovs") or []
        if fovs and fovs[0].get("frame_rate") is not None:
            return float(fovs[0]["frame_rate"])
    return None


def _frame_rate_v2(acquisition: dict) -> Optional[float]:
    """Frame rate from a v2 ImagingConfig.sampling_strategy.frame_rate."""
    for stream in acquisition.get("data_streams") or []:
        for config in stream.get("configurations") or []:
            if config.get("object_type") == _IMAGING_CONFIG_TYPE:
                sampling = config.get("sampling_strategy") or {}
                if sampling.get("frame_rate") is not None:
                    return float(sampling["frame_rate"])
    return None


def _frame_rate_platform(platform: dict) -> Optional[float]:
    """Frame rate from a v1 platform.json imaging_plane_groups."""
    groups = platform.get("imaging_plane_groups") or []
    if groups and groups[0].get("acquisition_framerate_Hz") is not None:
        return float(groups[0]["acquisition_framerate_Hz"])
    return None


def _optional_field(path: Optional[Path], key: str) -> Optional[str]:
    """Read one top-level key from an optional JSON file, else None."""
    if path is None:
        return None
    return _load_json(path).get(key)


def load_motion_correction_metadata(
    input_dir: Path,
) -> MotionCorrectionMetadata:
    """Read v1/v2 metadata for motion correction from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). ``frame_rate`` falls back to ``platform.json``
    when the core file lacks it; the caller applies any final CLI default.
    ``subject_id`` and ``dataset_name`` are read independently from
    ``subject.json`` / ``data_description.json`` (no cross-file
    validation). Raises FileNotFoundError if neither core file is found.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.

    Returns
    -------
    MotionCorrectionMetadata
        The minimal fields this capsule consumes.
    """
    input_dir = Path(input_dir)
    acq_path = _find(input_dir, "acquisition.json")
    session_path = _find(input_dir, "session.json")

    dataset_name = _optional_field(
        _find(input_dir, "data_description.json"), "name"
    )
    subject_id = _optional_field(
        _find(input_dir, "subject.json"), "subject_id"
    )

    if acq_path is not None:
        acquisition = _load_json(acq_path)
        frame_rate = _frame_rate_v2(acquisition)
        instrument_id = acquisition.get("instrument_id")
    elif session_path is not None:
        session = _load_json(session_path)
        frame_rate = _frame_rate_v1(session)
        instrument_id = session.get("rig_id")
    else:
        raise FileNotFoundError(
            "No acquisition.json (v2) or session.json (v1) found under "
            f"{input_dir}"
        )

    if frame_rate is None:
        platform_path = _find(input_dir, "platform.json")
        if platform_path is not None:
            frame_rate = _frame_rate_platform(_load_json(platform_path))

    return MotionCorrectionMetadata(
        frame_rate=frame_rate,
        instrument_id=instrument_id,
        subject_id=subject_id,
        dataset_name=dataset_name,
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
    return DataProcess(
        process_type=ProcessName.VIDEO_MOTION_CORRECTION,
        name="Suite2P motion correction",
        stage=ProcessStage.PROCESSING,
        code=Code(
            url=CODE_URL,
            name="aind-ophys-motion-correction",
            version=os.getenv("VERSION", ""),
            parameters=parameters,
            input_data=(
                [DataAsset(name=n) for n in input_data] if input_data else None
            ),
        ),
        experimenters=[],
        start_date_time=start_time.astimezone(_SEATTLE_TZ),
        end_date_time=end_time.astimezone(_SEATTLE_TZ),
        output_path=output_path,
        output_parameters=output_parameters or {},
    )


def build_processing(data_processes: List[DataProcess]) -> Processing:
    """Wrap data processes into a full v2 Processing object."""
    return Processing(data_processes=data_processes)


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
    output_dir = Path(output_dir)
    out_path = output_dir / "processing.json"
    with open(out_path, "w") as f:
        json.dump(json.loads(processing.model_dump_json()), f, indent=4)
    return out_path

"""quality_control.json generation for motion correction.

All QC lives here: metric builders return ``QCMetric`` objects, and the
entrypoint accumulates them into ONE full ``QualityControl`` document via
the collector pattern (never per-metric files). ``default_grouping`` is the
project-standard ``["evaluation"]`` so the QC aggregator's
``QualityControl.__add__`` merges cleanly.
"""

import json
from datetime import datetime as dt
from pathlib import Path
from typing import List

import pytz
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import DropdownMetric

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

DEFAULT_GROUPING = ["evaluation"]


def pending_qc_status() -> QCStatus:
    """Return a PENDING QCStatus timestamped to US/Pacific."""
    return QCStatus(
        evaluator="Pending review",
        status=Status.PENDING,
        timestamp=dt.now(_SEATTLE_TZ).isoformat(),
    )


def build_registration_summary_metric(
    unique_id: str,
    reference_filepath: str,
) -> QCMetric:
    """Construct the registration-summary QCMetric.

    Parameters
    ----------
    unique_id : str
        Plane/FOV id used in the metric name.
    reference_filepath : str
        Path to the registration-summary image, relative to the asset.

    Returns
    -------
    QCMetric
        The registration-summary metric (pending review).
    """
    return QCMetric(
        name=f"{unique_id} Registration Summary",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "Registration Summary", "type": "Operational QC"},
        description=(
            "Review the registration summary plot to ensure that the "
            "motion correction is accurate and sufficient."
        ),
        status_history=[pending_qc_status()],
        reference=reference_filepath,
        value=DropdownMetric(
            value="",
            options=[
                "Motion correction successful",
                "No motion correction applied",
                "Motion correction failed",
                "Motion correction partially successful",
            ],
            status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL],
        ),
    )


def build_fov_quality_metric(
    unique_id: str,
    reference_filepath: str,
) -> QCMetric:
    """Construct the FOV-quality QCMetric.

    Parameters
    ----------
    unique_id : str
        Plane/FOV id used in the metric name.
    reference_filepath : str
        Path to the combined projection image, relative to the asset.

    Returns
    -------
    QCMetric
        The FOV-quality metric (pending review).
    """
    return QCMetric(
        name=f"{unique_id} FOV Quality",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "FOV Quality", "type": "Operational QC"},
        description=(
            "Review the avg. and max. projections to ensure that the "
            "FOV quality is sufficient."
        ),
        status_history=[pending_qc_status()],
        reference=reference_filepath,
        value=DropdownMetric(
            value="Quality is sufficient",
            options=[
                "Quality is sufficient",
                "Timeseries shuffled between planes",
                "Field of view associated with incorrect area and/or depth",
                "Paired plane cross talk: Extreme",
                "Paired plane cross-talk: Moderate",
            ],
            status=[
                Status.PASS,
                Status.FAIL,
                Status.FAIL,
                Status.FAIL,
                Status.FAIL,
            ],
        ),
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return QualityControl(
        metrics=list(metrics),
        default_grouping=DEFAULT_GROUPING,
    )


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    output_dir = Path(output_dir)
    out_path = output_dir / "quality_control.json"
    with open(out_path, "w") as f:
        json.dump(json.loads(quality_control.model_dump_json()), f, indent=4)
    return out_path

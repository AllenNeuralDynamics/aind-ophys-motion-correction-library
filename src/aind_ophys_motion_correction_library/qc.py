"""quality_control.json generation for motion correction.

Motion-correction metric content over the shared ``aind_pophys_metadata``
QC builders: the metric skeleton, the PENDING status, the ``QualityControl``
wrapper with the project-standard ``default_grouping``, and the writer all
live in the shared library. ``job.run`` accumulates the metrics built here and
emits ONE full ``quality_control.json`` (never per-metric files).
"""

from pathlib import Path
from typing import List

from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Status,
)
from aind_pophys_metadata import quality_control as shared_qc

EVALUATION = "Motion Correction"


def pending_qc_status() -> QCStatus:
    """Return a PENDING QCStatus timestamped to US/Pacific."""
    return shared_qc.pending_qc_status()


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
    return shared_qc.dropdown_metric(
        name=f"{unique_id} Registration Summary",
        evaluation=EVALUATION,
        options=[
            "Motion correction successful",
            "No motion correction applied",
            "Motion correction failed",
            "Motion correction partially successful",
        ],
        status=[Status.PASS, Status.FAIL, Status.FAIL, Status.FAIL],
        reference=reference_filepath,
        description=(
            "Registration summary plot. Check that motion correction is "
            "accurate and sufficient."
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
        Path to the max-projection image, relative to the asset.

    Returns
    -------
    QCMetric
        The FOV-quality metric (pending review).
    """
    return shared_qc.dropdown_metric(
        name=f"{unique_id} FOV Quality",
        evaluation=EVALUATION,
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
        reference=reference_filepath,
        description=(
            "Maximum projection of the registered movie. Check that the "
            "field of view is in focus and usable."
        ),
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


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
    return shared_qc.write_quality_control_json(quality_control, output_dir)

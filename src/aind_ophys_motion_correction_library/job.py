"""Top-level motion-correction orchestration.

``run()`` is the library entry point called by the thin capsule wrapper:
read metadata (v1 or v2) -> configure logging -> prepare per-plane input ->
run Suite2P -> write the registered movie + diagnostics -> emit a full
``processing.json`` and ``quality_control.json``.
"""

import copy
import json
import logging
import shutil
import tempfile
from datetime import datetime as dt
from functools import partial
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import suite2p
from aind_ophys_utils.video_utils import encode_video
from aind_pophys_metadata import io
from log_schema.main import setup_logging
from PIL import Image

from . import qc
from . import registration as reg
from .settings import MotionCorrectionSettings
from .utils import metadata_utils as mu
from .utils.paths import relative_to_root

logger = logging.getLogger("Suite2P motion correction")

PROCESS_NAME = "aind-ophys-motion-correction"
_BERGAMO_INSTRUMENT = "Bergamo"

# Output filename suffixes appended to the plane basename.
_OUTPUT_SUFFIXES = (
    ("motion_corrected_output", "_registered.h5"),
    ("motion_diagnostics_output", "_motion_transform.csv"),
    ("max_projection_output", "_maximum_projection.png"),
    ("avg_projection_output", "_average_projection.png"),
    ("registration_summary_output", "_registration_summary.png"),
    ("motion_correction_preview_output", "_motion_preview.webm"),
    ("output_json", "_motion_correction_output.json"),
)


def _load_raw_metadata(input_dir: Path) -> dict:
    """Load the raw v1 session dict for the per-plane prep functions.

    The multiplane/single-plane preparation helpers still consume the v1
    ``session.json`` structure directly; returns an empty dict for v2-only
    inputs (no session.json present).
    """
    session_fp = io.find(input_dir, "session.json")
    return io.load_json(session_fp) if session_fp else {}


def _prepare_input(
    settings: MotionCorrectionSettings,
    metadata: mu.MotionCorrectionMetadata,
    session: dict,
    frame_rate_hz: Optional[float],
) -> dict:
    """Route by data_type to locate the per-plane input to register.

    Returns a context dict with input_file, unique_id, output_dir,
    frame_rate_hz, and reference_image_fp.
    """
    data_dir = settings.input_dir
    output_dir = settings.output_dir
    data_type = settings.data_type.lower()
    reference_image_fp = ""

    if data_type in ("tif", "bruker"):
        unique_id = "plane_0"
        try:
            input_file = next(data_dir.rglob("*/pophys"))
        except StopIteration:
            input_file = next(data_dir.rglob("pophys"))
        output_dir = reg.make_output_directory(output_dir, unique_id)
    elif data_type == "h5":
        unique_id = metadata.dataset_name or "plane_0"
        if _BERGAMO_INSTRUMENT in (metadata.instrument_id or ""):
            h5_file, output_dir, reference_image_fp = (
                reg.singleplane_motion_correction(
                    data_dir,
                    output_dir,
                    session,
                    unique_id,
                    debug=settings.debug,
                )
            )
        else:
            h5_file, output_dir, frame_rate_hz = (
                reg.multiplane_motion_correction(
                    data_dir, output_dir, debug=settings.debug
                )
            )
        input_file = str(h5_file)
    else:
        raise ValueError(
            f"Data type {settings.data_type} not supported. "
            "Please use 'tif', 'bruker', or 'h5'."
        )

    return {
        "input_file": input_file,
        "unique_id": unique_id,
        "output_dir": output_dir,
        "frame_rate_hz": frame_rate_hz,
        "reference_image_fp": reference_image_fp,
    }


def _build_output_paths(args: dict, output_dir: Path, basename: str) -> None:
    """Populate the per-output file paths in ``args`` (in place)."""
    for key, suffix in _OUTPUT_SUFFIXES:
        args[key] = str(Path(output_dir) / (basename.split(".")[0] + suffix))


def _build_suite2p_args(
    settings: MotionCorrectionSettings, ctx: dict, args: dict
) -> dict:
    """Assemble the Suite2P ops dict from settings + resolved input."""
    input_file = ctx["input_file"]
    suite2p_args = suite2p.default_ops()
    if settings.data_type == "h5":
        suite2p_args["h5py"] = str(input_file)
    else:
        suite2p_args["data_path"] = str(input_file)
        suite2p_args["look_one_level_down"] = True
        suite2p_args["tiff_list"] = [
            str(i) for i in Path(input_file).glob("*.tif*")
        ]
    suite2p_args["roidetect"] = False
    suite2p_args["do_registration"] = settings.do_registration
    suite2p_args["align_by_chan"] = settings.align_by_chan
    suite2p_args["reg_tif"] = False
    suite2p_args["nimg_init"] = 500
    suite2p_args["maxregshift"] = settings.maxregshift
    suite2p_args["maxregshiftNR"] = settings.maxregshiftNR
    suite2p_args["batch_size"] = settings.batch_size
    if suite2p_args.get("h5py", ""):
        suite2p_args["h5py_key"] = "data"
    suite2p_args["smooth_sigma"] = settings.smooth_sigma
    suite2p_args["smooth_sigma_time"] = settings.smooth_sigma_time
    suite2p_args["nonrigid"] = settings.nonrigid
    suite2p_args["block_size"] = settings.block_size
    suite2p_args["snr_thresh"] = settings.snr_thresh
    suite2p_args["refImg"] = args["refImg"]
    suite2p_args["force_refImg"] = args["force_refImg"]
    return suite2p_args


def _stage_and_trim(
    settings: MotionCorrectionSettings, suite2p_args: dict, args: dict
) -> None:
    """Copy S3 inputs locally, warn on dtype, and trim empty frames."""
    if suite2p_args.get("h5py", "") and reg.is_S3(suite2p_args["h5py"]):
        dst = "/scratch/" + Path(suite2p_args["h5py"]).name
        logger.info(f"copying {suite2p_args['h5py']} from S3 to {dst}")
        shutil.copy(suite2p_args["h5py"], dst)
        suite2p_args["h5py"] = dst
    if suite2p_args.get("tiff_list", ""):
        reg.check_and_warn_on_datatype(
            filepath=suite2p_args["tiff_list"][0],
            logger=logger.warning,
            filetype="tiff",
        )
    else:
        reg.check_and_warn_on_datatype(
            filepath=suite2p_args["h5py"],
            logger=logger.warning,
            filetype="h5",
            h5py_key=suite2p_args["h5py_key"],
        )
    if args["auto_remove_empty_frames"] and settings.data_type.lower() != (
        "bruker"
    ):
        if suite2p_args.get("tiff_list", ""):
            low, high = reg.find_movie_start_end_empty_frames(
                filepath=suite2p_args["tiff_list"], logger=logger.warning
            )
        else:
            low, high = reg.find_movie_start_end_empty_frames(
                filepath=suite2p_args["h5py"],
                h5py_key=suite2p_args["h5py_key"],
                logger=logger.warning,
            )
        args["trim_frames_start"] = low
        args["trim_frames_end"] = high
        logger.info(f"Found ({low}, {high}) empty frames at start/end.")


def _resolve_reference(
    settings: MotionCorrectionSettings,
    suite2p_args: dict,
    args: dict,
    reference_image_fp: str,
) -> tuple:
    """Apply the Bruker bypass and external/computed reference image."""
    if settings.data_type.lower() == "bruker":
        suite2p_args["force_refImg"] = False
    if suite2p_args["force_refImg"] and len(suite2p_args["refImg"]) == 0:
        suite2p_args, args = reg.update_suite2p_args_reference_image(
            suite2p_args, args
        )
    if reference_image_fp:
        suite2p_args, args = reg.update_suite2p_args_reference_image(
            suite2p_args, args, reference_image_fp=reference_image_fp
        )
    return suite2p_args, args


def _load_ops_and_clip(suite2p_args: dict, args: dict, tdir: str) -> dict:
    """Load Suite2P ops/binary and clip offset outliers.

    Returns a dict with data, ops, delta_x/y, clipped_indices, is_valid.
    """
    bin_path = list(Path(tdir).rglob("data.bin"))[0]
    ops_path = list(Path(tdir).rglob("ops.npy"))[0]
    ops = np.load(ops_path, allow_pickle=True).item()

    detrend_size = int(args["frame_rate_hz"] * args["outlier_detrend_window"])
    xlimit = int(ops["Lx"] * args["outlier_maxregshift"])
    ylimit = int(ops["Ly"] * args["outlier_maxregshift"])
    delta_x, x_clipped = reg.identify_and_clip_outliers(
        np.array(ops["xoff"]), detrend_size, xlimit
    )
    delta_y, y_clipped = reg.identify_and_clip_outliers(
        np.array(ops["yoff"]), detrend_size, ylimit
    )
    clipped_indices = list(set(x_clipped).union(set(y_clipped)))

    data = suite2p.io.BinaryFile(ops["Ly"], ops["Lx"], bin_path).data
    if args["clip_negative"]:
        data[data < 0] = 0
        data = np.uint16(data)
    if not suite2p_args["nonrigid"]:
        for frame_index in clipped_indices:
            dx = delta_x[frame_index] - ops["xoff"][frame_index]
            dy = delta_y[frame_index] - ops["yoff"][frame_index]
            data[frame_index] = suite2p.registration.rigid.shift_frame(
                data[frame_index], dy, dx
            )
    reg.reset_frame_shift(
        data,
        delta_y,
        delta_x,
        args["trim_frames_start"],
        args["trim_frames_end"],
    )
    is_valid = np.ones(len(data), dtype="bool")
    is_valid[: args["trim_frames_start"]] = False
    is_valid[len(data) - args["trim_frames_end"] :] = False
    return {
        "data": data,
        "ops": ops,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "clipped_indices": clipped_indices,
        "is_valid": is_valid,
    }


def _write_registered_h5(
    suite2p_args: dict, args: dict, reg_out: dict
) -> None:
    """Write the registered movie + metadata + reg_metrics to HDF5."""
    data = reg_out["data"]
    ops = reg_out["ops"]
    with h5py.File(args["motion_corrected_output"], "w") as f:
        f.create_dataset("data", data=data, chunks=(1, *data.shape[1:]))
        f.create_dataset("ref_image", data=suite2p_args["refImg"])
        args_copy = copy.deepcopy(args)
        suite_args_copy = copy.deepcopy(suite2p_args)
        suite_args_copy.pop("refImg")
        args_copy.pop("refImg")
        args_copy["suite2p_args"] = suite_args_copy
        f.create_dataset(
            name="metadata", data=json.dumps(args_copy).encode("utf-8")
        )
        f.create_group("reg_metrics")
        f.create_dataset("reg_metrics/regDX", data=ops.get("regDX", []))
        f.create_dataset("reg_metrics/regPC", data=ops.get("regPC", []))
        f.create_dataset("reg_metrics/tPC", data=ops.get("tPC", []))
    logger.info(f"saved Suite2P output to {args['motion_corrected_output']}")


def _write_motion_csv(suite2p_args: dict, args: dict, reg_out: dict) -> None:
    """Write the motion-offset CSV (nonrigid- or rigid-shaped)."""
    ops = reg_out["ops"]
    if suite2p_args["nonrigid"]:

        def _fmt(key):
            """Format a per-block ops array column as CSV-safe strings."""
            return [
                np.array2string(
                    arr,
                    separator=",",
                    suppress_small=True,
                    max_line_width=4096,
                )
                for arr in ops[key]
            ]

        df = pd.DataFrame(
            {
                "framenumber": list(range(ops["nframes"])),
                "x": ops["xoff"],
                "y": ops["yoff"],
                "x_pre_clip": ops["xoff"],
                "y_pre_clip": ops["yoff"],
                "correlation": ops["corrXY"],
                "is_valid": reg_out["is_valid"],
                "nonrigid_x": _fmt("xoff1"),
                "nonrigid_y": _fmt("yoff1"),
                "nonrigid_corr": _fmt("corrXY1"),
            }
        )
    else:
        df = pd.DataFrame(
            {
                "framenumber": list(range(ops["nframes"])),
                "x": reg_out["delta_x"],
                "y": reg_out["delta_y"],
                "x_pre_clip": ops["xoff"],
                "y_pre_clip": ops["yoff"],
                "correlation": ops["corrXY"],
                "is_valid": reg_out["is_valid"],
            }
        )
    df.to_csv(path_or_buf=args["motion_diagnostics_output"], index=False)


def _write_projections_and_previews(
    suite2p_args: dict, args: dict, reg_out: dict
) -> None:
    """Write projections, summary PNGs, and the preview movie."""
    data = reg_out["data"]
    mx_proj = reg.projection_process(data, projection="max")
    av_proj = reg.projection_process(data, projection="avg")
    for im, dst_path in zip(
        [mx_proj, av_proj],
        [args["max_projection_output"], args["avg_projection_output"]],
    ):
        with Image.fromarray(im) as pilim:
            pilim.save(dst_path)

    df = pd.read_csv(args["motion_diagnostics_output"])
    reg.make_png(
        Path(args["max_projection_output"]),
        Path(args["avg_projection_output"]),
        df,
        Path(args["registration_summary_output"]),
    )
    if "nonrigid_x" in df.keys():
        p = Path(args["registration_summary_output"])
        reg.make_nonrigid_png(
            Path(args["motion_corrected_output"]),
            Path(args["avg_projection_output"]),
            df,
            p.parent.joinpath(p.stem + "_nonrigid" + p.suffix),
        )

    ds_partial = partial(
        reg.downsample_normalize,
        frame_rate=args["frame_rate_hz"],
        bin_size=args["preview_frame_bin_seconds"],
        lower_quantile=args["movie_lower_quantile"],
        upper_quantile=args["movie_upper_quantile"],
    )
    if suite2p_args.get("h5py", ""):
        raw = Path(suite2p_args["h5py"])
    else:
        raw = reg.tiff_to_numpy(suite2p_args["tiff_list"])
    processed = [
        ds_partial(i) for i in [raw, Path(args["motion_corrected_output"])]
    ]
    try:
        tiled = np.block(processed)
        fps = (
            args["preview_playback_factor"] / args["preview_frame_bin_seconds"]
        )
        encode_video(tiled, args["motion_correction_preview_output"], fps)
    except Exception:
        logger.info("Could not write motion correction preview")


def _write_reg_metrics(suite2p_args: dict, args: dict, reg_out: dict) -> None:
    """Append crispness + residual optical flow and write PC flow PNGs."""
    mc_out = args["motion_corrected_output"]
    if suite2p_args.get("h5py", ""):
        raw_src = h5py.File(suite2p_args["h5py"])
        raw_mov = raw_src["data"]
    else:
        raw_src = None
        raw_mov = reg.tiff_to_numpy(suite2p_args["tiff_list"])
    with h5py.File(mc_out, "r+") as f:
        reg_dx = f["reg_metrics/regDX"][:]
        crispness = reg.compute_crispness(raw_mov, f["data"])
        farneback_dx = np.array([])
        if f["reg_metrics/regPC"][:].any():
            flows, farneback_dx = reg.compute_residual_optical_flow(
                f["reg_metrics/regPC"]
            )
            f.create_dataset("reg_metrics/farnebackROF", data=flows)
            f.create_dataset("reg_metrics/farnebackDX", data=farneback_dx)
        f.create_dataset("reg_metrics/crispness", data=crispness)
    if raw_src is not None:
        raw_src.close()
    if reg_dx.any() and farneback_dx.any():
        for iPC in set(
            (np.argmax(reg_dx[:, -1]), np.argmax(farneback_dx[:, -1]))
        ):
            p = Path(args["registration_summary_output"])
            reg.flow_png(Path(mc_out), str(p.parent / p.stem), iPC)


def _write_quality_control(
    settings: MotionCorrectionSettings, ctx: dict, args: dict
) -> None:
    """Write one full quality_control.json for this plane."""
    results_root = Path(settings.output_dir)
    output_dir = Path(ctx["output_dir"])
    summary_ref = relative_to_root(
        results_root, Path(args["registration_summary_output"])
    )
    combined_ref = relative_to_root(
        results_root, Path(args["max_projection_output"])
    )
    metrics = [
        qc.build_registration_summary_metric(ctx["unique_id"], summary_ref),
        qc.build_fov_quality_metric(ctx["unique_id"], combined_ref),
    ]
    qc.write_quality_control_json(
        qc.build_quality_control(metrics), output_dir
    )


def _write_processing_metadata(
    settings: MotionCorrectionSettings,
    ctx: dict,
    start_time: dt,
    error: Optional[BaseException],
) -> None:
    """Best-effort writer for processing.json. Never raises.

    Failures here must not mask a registration failure already propagating
    out of ``run()``. Falls back to the results root when the run failed
    before the per-plane output directory was resolved.
    """
    results_root = Path(settings.output_dir).resolve()
    output_dir = Path(ctx.get("output_dir") or results_root)
    try:
        args_for_code = copy.deepcopy(ctx.get("args") or {})
        args_for_code.pop("refImg", None)
        data_process = mu.build_data_process(
            parameters=args_for_code,
            start_time=start_time,
            end_time=dt.now(),
            input_data=ctx.get("input_data"),
            output_path=relative_to_root(results_root, output_dir),
            plane_id=ctx.get("unique_id"),
            error=error,
        )
        mu.write_processing_json(
            mu.build_processing([data_process]), output_dir
        )
    except Exception as proc_exc:
        logger.error(
            f"processing.json write failed (non-fatal): {proc_exc}"
        )


def _verify_outputs(output_dir: Path) -> None:
    """Round-trip the emitted metadata through the core v2 schema objects."""
    from aind_data_schema.core.processing import Processing
    from aind_data_schema.core.quality_control import QualityControl

    Processing.model_validate_json(
        (Path(output_dir) / "processing.json").read_text()
    )
    QualityControl.model_validate_json(
        (Path(output_dir) / "quality_control.json").read_text()
    )


def _run_motion_correction(
    settings: MotionCorrectionSettings,
    metadata: mu.MotionCorrectionMetadata,
    ctx: dict,
) -> None:
    """Register the movie and write every output except processing.json.

    ``ctx`` is populated as the run progresses so the ``processing.json``
    writer in ``run()``'s ``finally`` can report partial provenance even
    when this raises.
    """
    session = _load_raw_metadata(settings.input_dir)
    ctx.update(
        _prepare_input(settings, metadata, session, metadata.frame_rate)
    )
    if ctx["frame_rate_hz"] is None:
        ctx["frame_rate_hz"] = settings.frame_rate
        logger.warning(
            "Using CLI fallback frame rate: %s", settings.frame_rate
        )
    if ctx["frame_rate_hz"] is None:
        raise ValueError(
            "frame_rate could not be resolved from metadata or --frame_rate"
        )
    ctx["input_data"] = [
        relative_to_root(
            Path(settings.input_dir).resolve(), Path(ctx["input_file"])
        )
    ]

    args = settings.model_dump()
    args["input_dir"] = str(settings.input_dir)
    args["output_dir"] = str(settings.output_dir)
    args["tmp_dir"] = str(settings.tmp_dir)
    args["frame_rate_hz"] = ctx["frame_rate_hz"]
    args["refImg"] = (
        [ctx["reference_image_fp"]] if ctx["reference_image_fp"] else []
    )
    ctx["args"] = args

    basename = (
        ctx["unique_id"]
        if settings.data_type.lower() in ("tif", "bruker")
        else Path(ctx["input_file"]).name
    )
    _build_output_paths(args, ctx["output_dir"], basename)

    suite2p_args = _build_suite2p_args(settings, ctx, args)
    _stage_and_trim(settings, suite2p_args, args)
    suite2p_args, args = _resolve_reference(
        settings, suite2p_args, args, ctx["reference_image_fp"]
    )
    ctx["args"] = args

    tmp_dir = tempfile.TemporaryDirectory(dir=settings.tmp_dir)
    suite2p_args["save_path0"] = tmp_dir.name
    suite2p.run_s2p(suite2p_args)

    reg_out = _load_ops_and_clip(suite2p_args, args, tmp_dir.name)
    _write_registered_h5(suite2p_args, args, reg_out)
    _write_motion_csv(suite2p_args, args, reg_out)
    _write_projections_and_previews(suite2p_args, args, reg_out)
    _write_reg_metrics(suite2p_args, args, reg_out)
    _write_quality_control(settings, ctx, args)

    try:
        tmp_dir.cleanup()
    except Exception:
        shutil.rmtree(tmp_dir.name, ignore_errors=True)


def run(settings: Optional[MotionCorrectionSettings] = None) -> None:
    """Run Suite2P motion correction end to end.

    ``processing.json`` is written from a ``finally`` block so a failed run
    still leaves a document describing what was attempted, with the error
    recorded in ``notes``.

    Parameters
    ----------
    settings : MotionCorrectionSettings, optional
        Parsed settings; constructed from CLI/env when omitted. When
        ``settings.verify`` is True, the emitted metadata is round-tripped
        through the core v2 schema objects after writing.
    """
    settings = settings or MotionCorrectionSettings()
    start_time = dt.now()
    metadata = mu.load_motion_correction_metadata(settings.input_dir)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    logger.info(
        "Starting motion correction", extra={"event_type": "stage_start"}
    )

    ctx: dict = {}
    run_error: Optional[BaseException] = None
    try:
        _run_motion_correction(settings, metadata, ctx)
        logger.info(
            "Motion correction complete",
            extra={"event_type": "stage_complete"},
        )
    except BaseException as exc:
        run_error = exc
        logger.error(
            "Motion correction failed", extra={"event_type": "stage_error"}
        )
        logger.exception("Unhandled exception")
        raise
    finally:
        _write_processing_metadata(settings, ctx, start_time, run_error)

    if settings.verify:
        _verify_outputs(ctx["output_dir"])

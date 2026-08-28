"""Top-level motion-correction orchestration.

``run()`` is the library entry point called by the thin capsule wrapper:
read metadata (v1 or v2) -> configure logging -> resolve this task's plane
identity -> run Suite2P -> write the registered movie + diagnostics -> emit a
full ``processing.json`` and ``quality_control.json``.
"""

import copy
import json
import logging
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import suite2p
from aind_data_schema_models.process_names import ProcessName
from aind_ophys_utils.video_utils import encode_video
from aind_pophys_metadata import CoreMetadata, find_one, io, relative_to_root
from aind_pophys_metadata.processing import collect_upstream_process_names
from aind_pophys_metadata.runner import stage_guard
from log_schema.main import setup_logging
from PIL import Image

from . import qc
from . import registration as reg
from .settings import MotionCorrectionSettings
from .utils import metadata_utils as mu

logger = logging.getLogger("Suite2P motion correction")

PROCESS_NAME = "aind-ophys-motion-correction"
DATA_PROCESS_NAME = "Suite2P motion correction"
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


def _is_bergamo(metadata: mu.MotionCorrectionMetadata) -> bool:
    """Return whether this acquisition came off a Bergamo rig.

    Parameters
    ----------
    metadata : mu.MotionCorrectionMetadata
        The acquisition metadata view.

    Returns
    -------
    bool
        True for a Bergamo single-plane acquisition.
    """
    return _BERGAMO_INSTRUMENT in (metadata.instrument_id or "")


def _resolve_plane_input(
    settings: MotionCorrectionSettings,
    metadata: mu.MotionCorrectionMetadata,
    core: CoreMetadata,
) -> dict:
    """Locate this task's movie and resolve its canonical plane identity.

    Runs before the metadata guard, which needs the per-plane output
    directory and plane id up front. Only discovery happens here; anything
    that reads or writes movie data belongs in :func:`_prepare_movie`.

    Parameters
    ----------
    settings : MotionCorrectionSettings
        Parsed settings.
    metadata : mu.MotionCorrectionMetadata
        The acquisition metadata view.
    core : aind_pophys_metadata.CoreMetadata
        The located acquisition metadata document.

    Returns
    -------
    dict
        Context with ``input_file``, ``unique_id``, ``output_dir``,
        ``frame_rate_hz`` and ``reference_image_fp``.

    Raises
    ------
    ValueError
        If the data type is unsupported, or no source supplies a frame rate.
    """
    data_dir = settings.input_dir
    results_root = settings.output_dir
    data_type = settings.data_type.lower()
    frame_rate_hz = metadata.frame_rate

    if data_type in ("tif", "bruker"):
        unique_id = mu.resolve_plane_id(metadata.fov_ids)
        input_file = find_one(
            data_dir, "*/pophys", required=False
        ) or find_one(data_dir, "pophys")
        output_dir = reg.make_output_directory(results_root, unique_id)
    elif data_type == "h5" and _is_bergamo(metadata):
        # The Bergamo converter names its stitched movie after the plane id
        # (e.g. plane_0/plane_0.h5) -- the same id this capsule uses to name
        # its own output directory -- so one value serves as both.
        unique_id = mu.resolve_plane_id(
            metadata.fov_ids, fallback=mu.DEFAULT_PLANE_ID
        )
        input_file, output_dir = reg.find_singleplane_input(
            data_dir, results_root, unique_id, unique_id
        )
    elif data_type == "h5":
        input_file, directory_id = reg.find_multiplane_input(data_dir)
        unique_id = mu.resolve_plane_id(
            metadata.fov_ids, directory_id=directory_id
        )
        output_dir = reg.make_output_directory(results_root, unique_id)
        frame_rate_hz = reg.resolve_multiplane_frame_rate(
            data_dir, core, settings.frame_rate
        )
    else:
        raise ValueError(
            f"Data type {settings.data_type} not supported. "
            "Please use 'tif', 'bruker', or 'h5'."
        )

    if frame_rate_hz is None:
        raise ValueError(
            "No frame rate could be resolved from "
            f"{Path(core.core_path).name}, {io.PLATFORM_FILE} or "
            "--frame_rate."
        )
    return {
        "input_file": input_file,
        "unique_id": unique_id,
        "output_dir": output_dir,
        "frame_rate_hz": frame_rate_hz,
        "reference_image_fp": "",
    }


def _prepare_movie(
    settings: MotionCorrectionSettings,
    metadata: mu.MotionCorrectionMetadata,
    core: CoreMetadata,
    ctx: dict,
) -> None:
    """Build the reference movie and apply any debug truncation, in place.

    Parameters
    ----------
    settings : MotionCorrectionSettings
        Parsed settings.
    metadata : mu.MotionCorrectionMetadata
        The acquisition metadata view.
    core : aind_pophys_metadata.CoreMetadata
        The located acquisition metadata document.
    ctx : dict
        Context from :func:`_resolve_plane_input`; ``input_file`` and
        ``reference_image_fp`` are updated.
    """
    if settings.data_type.lower() != "h5":
        return
    if _is_bergamo(metadata):
        input_file, _, reference_image_fp = reg.singleplane_motion_correction(
            Path(ctx["input_file"]),
            ctx["output_dir"],
            core,
            debug=settings.debug,
            scratch_dir=settings.tmp_dir,
        )
        ctx["input_file"] = input_file
        ctx["reference_image_fp"] = reference_image_fp
    elif settings.debug:
        ctx["input_file"] = reg.truncate_for_debug(
            Path(ctx["input_file"]),
            ctx["unique_id"],
            ctx["frame_rate_hz"],
            scratch_dir=settings.tmp_dir,
        )


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
    bin_path = find_one(tdir, "data.bin")
    ops_path = find_one(tdir, "ops.npy")
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

        def _fmt(key: str) -> List[str]:
            """Format a per-block ops array column as CSV-safe strings.

            The line width is unbounded so a wide block grid stays on one
            line; a wrapped array splits the CSV row it belongs to.
            """
            return [
                np.array2string(
                    arr,
                    separator=",",
                    suppress_small=True,
                    max_line_width=sys.maxsize,
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
        logger.warning(
            "Could not write motion correction preview", exc_info=True
        )


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
    max_proj_ref = relative_to_root(
        results_root, Path(args["max_projection_output"])
    )
    metrics = [
        qc.build_registration_summary_metric(ctx["unique_id"], summary_ref),
        qc.build_fov_quality_metric(ctx["unique_id"], max_proj_ref),
    ]
    qc.write_quality_control_json(
        qc.build_quality_control(metrics), output_dir
    )


def _registration_output_parameters(args: dict, reg_out: dict) -> dict:
    """Scalar summary statistics describing the registered movie.

    ``output_parameters`` carries statistics about a step's *outputs*, chosen
    so a reviewer can triage the registration from the metadata alone without
    opening the movie. Input configuration belongs in ``code.parameters`` and
    is deliberately not repeated here.

    Parameters
    ----------
    args : dict
        Resolved run arguments, read for the trim counts actually applied.
    reg_out : dict
        Registration result from :func:`_load_ops_and_clip`.

    Returns
    -------
    dict
        The summary statistics.
    """
    delta_x = np.abs(np.asarray(reg_out["delta_x"], dtype=float))
    delta_y = np.abs(np.asarray(reg_out["delta_y"], dtype=float))
    summary = {
        "frame_count": int(len(reg_out["data"])),
        "valid_frame_count": int(np.count_nonzero(reg_out["is_valid"])),
        "frames_trimmed_start": int(args["trim_frames_start"]),
        "frames_trimmed_end": int(args["trim_frames_end"]),
        "outlier_frames_clipped": int(len(reg_out["clipped_indices"])),
    }
    if delta_x.size and delta_y.size:
        summary.update(
            {
                "max_abs_shift_x_pixels": float(delta_x.max()),
                "max_abs_shift_y_pixels": float(delta_y.max()),
                "mean_abs_shift_x_pixels": float(delta_x.mean()),
                "mean_abs_shift_y_pixels": float(delta_y.mean()),
            }
        )
    return summary


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


def _build_args(settings: MotionCorrectionSettings, ctx: dict) -> dict:
    """Assemble the resolved run arguments recorded as code parameters.

    Parameters
    ----------
    settings : MotionCorrectionSettings
        Parsed settings.
    ctx : dict
        Context from :func:`_resolve_plane_input`.

    Returns
    -------
    dict
        The run arguments, including the per-output file paths.
    """
    args = settings.model_dump()
    args["input_dir"] = str(settings.input_dir)
    args["output_dir"] = str(settings.output_dir)
    args["tmp_dir"] = str(settings.tmp_dir)
    args["frame_rate_hz"] = ctx["frame_rate_hz"]
    args["refImg"] = []
    basename = (
        ctx["unique_id"]
        if settings.data_type.lower() in ("tif", "bruker")
        else Path(ctx["input_file"]).name
    )
    _build_output_paths(args, ctx["output_dir"], basename)
    return args


def _run_motion_correction(
    settings: MotionCorrectionSettings,
    metadata: mu.MotionCorrectionMetadata,
    core: CoreMetadata,
    ctx: dict,
) -> dict:
    """Register the movie and write every output except processing.json.

    Parameters
    ----------
    settings : MotionCorrectionSettings
        Parsed settings.
    metadata : mu.MotionCorrectionMetadata
        The acquisition metadata view.
    core : aind_pophys_metadata.CoreMetadata
        The located acquisition metadata document.
    ctx : dict
        Context from :func:`_resolve_plane_input`, carrying ``args``.

    Returns
    -------
    dict
        Summary statistics for ``DataProcess.output_parameters``.
    """
    _prepare_movie(settings, metadata, core, ctx)
    args = ctx["args"]
    args["refImg"] = (
        [ctx["reference_image_fp"]] if ctx["reference_image_fp"] else []
    )

    suite2p_args = _build_suite2p_args(settings, ctx, args)
    _stage_and_trim(settings, suite2p_args, args)
    suite2p_args, args = _resolve_reference(
        settings, suite2p_args, args, ctx["reference_image_fp"]
    )

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
    return _registration_output_parameters(args, reg_out)


def run(settings: Optional[MotionCorrectionSettings] = None) -> None:
    """Run Suite2P motion correction end to end.

    The registration is wrapped in the shared ``stage_guard``, which stamps
    the timings, emits the stage markers and writes ``processing.json``
    whether or not the body raises.

    Parameters
    ----------
    settings : MotionCorrectionSettings, optional
        Parsed settings; constructed from CLI/env when omitted. When
        ``settings.verify`` is True, the emitted metadata is round-tripped
        through the core v2 schema objects after writing.
    """
    settings = settings or MotionCorrectionSettings()
    core = CoreMetadata.load(settings.input_dir)
    metadata = mu.load_motion_correction_metadata(
        settings.input_dir,
        core=core,
        cli_override=settings.frame_rate,
    )
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )

    ctx = _resolve_plane_input(settings, metadata, core)
    ctx["args"] = _build_args(settings, ctx)
    ctx["input_data"] = [
        relative_to_root(
            Path(settings.input_dir).resolve(), Path(ctx["input_file"])
        )
    ]
    code_parameters = copy.deepcopy(ctx["args"])
    code_parameters.pop("refImg", None)

    results_root = Path(settings.output_dir).resolve()
    with stage_guard(
        ctx["output_dir"],
        ProcessName.VIDEO_MOTION_CORRECTION,
        mu.build_code(code_parameters, ctx["input_data"]),
        plane_id=ctx["unique_id"],
        logger=logger,
        name=DATA_PROCESS_NAME,
        output_path=relative_to_root(results_root, ctx["output_dir"]),
    ) as stage:
        stage.upstream_names = collect_upstream_process_names(
            settings.input_dir,
            exclude=f"{ctx['unique_id']}: {DATA_PROCESS_NAME}",
        )
        stage.output_parameters = _run_motion_correction(
            settings, metadata, core, ctx
        )

    if settings.verify:
        _verify_outputs(ctx["output_dir"])

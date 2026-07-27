"""CLI/environment settings for Suite2P motion correction.

Kept import-light (pydantic + stdlib only) so it stays cheap to import
and free of aind-data-schema / suite2p side effects.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class MotionCorrectionSettings(BaseSettings, cli_parse_args=True):
    """Settings for Suite2P motion correction.

    Configuration parameters for motion correction using Suite2P.
    Values can be provided via constructor, environment variables, or
    a ``.env`` file.
    """

    input_dir: Path = Field(
        default=Path("/data/"),
        description="File or directory where the h5 file is stored",
    )
    output_dir: Path = Field(
        default=Path("/results/"), description="Output directory"
    )
    debug: bool = Field(default=False, description="Run with partial dataset")
    verify: bool = Field(
        default=False,
        description="Round-trip the emitted processing.json / "
        "quality_control.json through the core v2 schema objects after "
        "writing, so bad output fails in-capsule rather than downstream",
    )
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description="Directory for temporary files produced by Suite2P",
    )
    data_type: str = Field(default="h5", description="bruker, tif, or h5")
    frame_rate: Optional[float] = Field(
        default=None,
        description="Frame rate [Hz] fallback used only when it cannot be "
        "read from acquisition/session/platform metadata",
    )
    do_registration: bool = Field(
        default=True,
        description="whether to register data (2 forces re-registration)",
    )
    batch_size: int = Field(
        default=500, description="Number of frames per batch"
    )
    align_by_chan: int = Field(
        default=1,
        description="when multi-channel, align by non-functional channel "
        "(1-based)",
    )
    maxregshift: float = Field(
        default=0.1,
        description="max allowed registration shift, as a fraction of frame "
        "max(width and height). Ignored if force_refImg is True",
    )
    force_refImg: bool = Field(
        default=True,
        description="Force the use of an external reference image",
    )
    nonrigid: bool = Field(
        default=True, description="Whether to use non-rigid registration"
    )
    block_size: list = Field(
        default_factory=lambda: [128, 128],
        description="Block size for non-rigid registration. Accepts a CLI/env "
        "string like '[128, 128]'; elements are coerced to int.",
    )
    snr_thresh: float = Field(
        default=1.2,
        description="if any nonrigid block is below this threshold, it gets "
        "smoothed until above it. 1.0 results in no smoothing",
    )
    maxregshiftNR: int = Field(
        default=5,
        description="maximum pixel shift allowed for nonrigid, relative to "
        "rigid",
    )
    outlier_detrend_window: float = Field(
        default=3.0,
        description="For outlier rejection in the xoff/yoff outputs of "
        "suite2p, the offsets are first de-trended with a median filter of "
        "this duration [seconds]. This is ~30 or 90 samples for 11 and 31 Hz "
        "sampling rates respectively.",
    )
    outlier_maxregshift: float = Field(
        default=0.05,
        description="Units [fraction FOV dim]. After median-filter "
        "detrending, outliers more than this value are clipped to it in x "
        "and y offset, independently. Similar to Suite2P's internal "
        "maxregshift but allows for low-frequency drift. Default 0.05 "
        "typically clips outliers to 512 * 0.05 = 25 pixels above or below "
        "the median trend.",
    )
    clip_negative: bool = Field(
        default=False,
        description="Whether to clip negative pixel values in output. Pixel "
        "values in raw movies are set by the current off a photomultiplier "
        "tube, so there can be negative pixels (current has a sign), possibly "
        "from rig noise. Some segmentation algorithms cannot handle negative "
        "values, so this sets those pixels to zero.",
    )
    max_reference_iterations: int = Field(
        default=8,
        description="Maximum number of iterations for creating a reference "
        "image",
    )
    auto_remove_empty_frames: bool = Field(
        default=True,
        description="Automatically detect empty noise frames at the start and "
        "end of the movie. Overrides trim_frames_start and trim_frames_end. "
        "Some movies contain empty pure-noise frames that receive large "
        "random shifts and throw off motion-border calculation; detecting "
        "them removes them from reference creation, smoothing parameter "
        "searches, and the motion-border calculation. The frames are still "
        "written, but any estimated shift is set to 0.",
    )
    trim_frames_start: int = Field(
        default=0,
        description="Number of frames to remove from the start of the movie "
        "if known. Removes frames from motion-border calculation and resets "
        "the frame shifts found. Frames are still written. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    trim_frames_end: int = Field(
        default=0,
        description="Number of frames to remove from the end of the movie if "
        "known. Removes frames from motion-border calculation and resets the "
        "frame shifts found. Frames are still written. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_end > 0",
    )
    do_optimize_motion_params: bool = Field(
        default=False,
        description="Search for the best smooth_sigma and smooth_sigma_time. "
        "Adds significant runtime and should only be run once per experiment, "
        "storing the resulting parameters for later use.",
    )
    smooth_sigma_time: int = Field(
        default=0,
        description="gaussian smoothing in time. Overridden if "
        "do_optimize_motion_params is set",
    )
    smooth_sigma: float = Field(
        default=1.15,
        description="~1 good for 2P recordings, recommend 3-5 for 1P. "
        "Overridden if do_optimize_motion_params is set",
    )
    use_ave_image_as_reference: bool = Field(
        default=False,
        description="Only if do_optimize_motion_params is set. After the best "
        "smoothing parameters are found, use the resulting average image as "
        "the reference for the full registration. Enables two-step "
        "registration when smooth_sigma_min=smooth_sigma_max and "
        "smooth_sigma_time_min=smooth_sigma_time_max and steps=1.",
    )
    movie_lower_quantile: float = Field(
        default=0.1,
        description="Lower quantile for avg-projection histogram adjustment "
        "of the movie",
    )
    movie_upper_quantile: float = Field(
        default=0.999,
        description="Upper quantile for avg-projection histogram adjustment "
        "of the movie",
    )
    preview_frame_bin_seconds: float = Field(
        default=2.0,
        description="Before creating the webm, movies are averaged into bins "
        "of this many seconds",
    )
    preview_playback_factor: float = Field(
        default=10.0,
        description="The preview movie plays back at this factor times "
        "real-time",
    )
    n_batches: int = Field(
        default=20,
        description="Number of batches to load from the movie for smoothing "
        "parameter testing. Batches are evenly spaced throughout the movie.",
    )
    smooth_sigma_min: float = Field(
        default=0.65,
        description="Minimum of the parameter search for smooth_sigma",
    )
    smooth_sigma_max: float = Field(
        default=2.15,
        description="Maximum of the parameter search for smooth_sigma",
    )
    smooth_sigma_steps: int = Field(
        default=4,
        description="Number of steps to grid between smooth_sigma and "
        "smooth_sigma_max",
    )
    smooth_sigma_time_min: float = Field(
        default=0,
        description="Minimum of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_max: float = Field(
        default=6,
        description="Maximum of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_steps: int = Field(
        default=7,
        description="Number of steps to grid between smooth_sigma_time and "
        "smooth_sigma_time_max. Large values add significant time.",
    )

    @field_validator("block_size", mode="before")
    @classmethod
    def _coerce_block_size(cls, value):
        """Coerce a CLI/env block_size to a list of ints.

        Accepts a JSON string (``"[128, 128]"``) or a list whose elements
        may be strings, and returns a list of ints. Kept a plain ``list``
        field (not ``List[int]``) so the auto-generated CO app-panel treats
        it as a single string parameter rather than one integer.
        """
        if isinstance(value, str):
            value = json.loads(value)
        return [int(x) for x in value]

    class Config:
        """Pydantic settings config: env prefix + .env support."""

        env_prefix = "MOTION_CORRECTION_"
        case_sensitive = False
        env_file = ".env"

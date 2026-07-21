"""Suite2P motion-correction algorithm and per-plane data preparation.

Ported from the aind-ophys-motion-correction capsule. Pure computation and
file I/O only; schema (processing.json) lives in ``utils.metadata_utils``,
QC in ``qc``, settings in ``settings``, and orchestration in ``job``.
"""

import json
import logging
import re
import subprocess
import warnings
from functools import lru_cache
from itertools import product
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import time
from typing import Callable, List, Optional, Tuple, Union

import cv2
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
from aind_ophys_utils.array_utils import normalize_array
from aind_ophys_utils.summary_images import mean_image
from aind_ophys_utils.video_utils import (
    downsample_array,
    downsample_h5_video,
)
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont
from ScanImageTiffReader import ScanImageTiffReader
from scipy.ndimage import median_filter
from scipy.stats import sigmaclip
from suite2p.registration.nonrigid import make_blocks
from suite2p.registration.register import (
    pick_initial_reference,
    register_frames,
)
from suite2p.registration.rigid import (
    apply_masks,
    compute_masks,
    phasecorr,
    phasecorr_reference,
    shift_frame,
)

from .sync_dataset import Sync

mpl.use("Agg")

logger = logging.getLogger(__name__)


def is_S3(file_path: str):
    """Test if a file is in a S3 bucket
    Parameters
    ----------
    file_path : str
        Location of the file.
    """
    return "s3fs" in subprocess.check_output(
        "df " + file_path + "| sed -n '2 p'", shell=True, text=True
    )


def h5py_byteorder_name(
    h5py_file: h5py.File, h5py_key: str
) -> Tuple[str, str]:
    """Get the byteorder and name of the dataset in the h5py file.

    Parameters
    ----------
    h5py_file : h5py.File
        h5py file object
    h5py_key : str
        key to the dataset

    Returns
    -------
    str
        byteorder of the dataset
    str
        name of the dataset
    """
    with h5py.File(h5py_file, "r") as f:
        byteorder = f[h5py_key].dtype.byteorder
        name = f[h5py_key].dtype.name
    return byteorder, name


def tiff_byteorder_name(tiff_file: Path) -> Tuple[str, str]:
    """Get the byteorder and name of the dataset in the tiff file.

    Parameters
    ----------
    tiff_file : Path
        Location of the tiff file

    Returns
    -------
    str
        byteorder of the dataset
    str
        name of the dataset
    """
    with ScanImageTiffReader(tiff_file) as reader:
        byteorder = reader.data().dtype.byteorder
        name = reader.data().dtype.name
    return byteorder, name


def h5py_to_numpy(
    h5py_file: str,
    h5py_key: str,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Converts a h5py dataset to a numpy array

    Parameters
    ----------
    h5py_file: str
        h5py file path
    h5py_key : str
        key to the dataset
    trim_frames_start : int
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_end : int
        Number of frames to disregard from the end of the movie. Default 0.
    Returns
    -------
    np.ndarray
        numpy array
    """
    with h5py.File(h5py_file, "r") as f:
        n_frames = f[h5py_key].shape[0]
        if trim_frames_start > 0 or trim_frames_end > 0:
            return f[h5py_key][trim_frames_start : n_frames - trim_frames_end]
        else:
            return f[h5py_key][:]


@lru_cache(maxsize=None)
def _tiff_to_numpy(tiff_file: Path) -> np.ndarray:
    """Read one TIFF file into a numpy array via ScanImageTiffReader."""
    with ScanImageTiffReader(tiff_file) as reader:
        return reader.data()


def tiff_to_numpy(
    tiff_list: List[Path], trim_frames_start: int = 0, trim_frames_end: int = 0
) -> np.ndarray:
    """
    Converts a list of TIFF files to a single numpy array, with optional frame trimming.  # noqa: E501

    Parameters
    ----------
    tiff_list : List[str]
        List of TIFF file paths to process
    trim_frames_start : int, optional
        Number of frames to remove from the start (default: 0)
    trim_frames_end : int, optional
        Number of frames to remove from the end (default: 0)

    Returns
    -------
    np.ndarray
        Combined array of all TIFF data with specified trimming

    Raises
    ------
    ValueError
        If trim values exceed total number of frames or are negative
    """
    if trim_frames_start < 0 or trim_frames_end < 0:
        raise ValueError("Trim values must be non-negative")

    def get_total_frames(tiff_files: List[Path]) -> int:
        """Calculate total number of frames across all TIFF files."""
        total = 0
        for tiff in tiff_files:
            with ScanImageTiffReader(tiff) as reader:
                total += reader.shape()[0]
        return total

    # Validate trim parameters
    total_frames = get_total_frames(tiff_list)
    if trim_frames_start + trim_frames_end >= total_frames:
        raise ValueError(
            f"Invalid trim values: start ({trim_frames_start}) + end ({trim_frames_end}) "  # noqa: E501
            f"must be less than total frames ({total_frames})"
        )

    # Initialize variables for frame counting
    processed_frames = 0
    arrays_to_stack = []

    for tiff_path in tiff_list:
        with ScanImageTiffReader(tiff_path) as reader:
            current_frames = reader.shape()[0]
        start_idx = max(0, trim_frames_start - processed_frames)
        end_idx = current_frames

        if processed_frames + current_frames > total_frames - trim_frames_end:
            end_idx = total_frames - trim_frames_end - processed_frames

        if start_idx < end_idx:  # Only process if there are frames to include
            data = np.array(_tiff_to_numpy(tiff_path)[start_idx:end_idx])
            arrays_to_stack.append(data)

        processed_frames += current_frames

        # Break if we've processed all needed frames
        if processed_frames >= total_frames - trim_frames_end:
            break

    # Stack all arrays along the appropriate axis
    if not arrays_to_stack:
        raise ValueError("No frames remained after trimming")

    return (
        np.concatenate(arrays_to_stack, axis=0)
        if len(arrays_to_stack) > 1
        else arrays_to_stack[0]
    )


def load_initial_frames(
    file_path: Union[str, list],
    h5py_key: str,
    n_frames: int,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Load a subset of frames from the hdf5 data specified by file_path.

    Only loads frames between trim_frames_start and n_frames - trim_frames_end
    from the movie. If both are 0, load frames from the full movie.

    Parameters
    ----------
    file_path : str
        Location of the raw ophys, HDF5 data to load.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_frames : int
        Number of frames to load from the input HDF5 data.

    Returns
    -------
    frames : array-like, (n_frames, nrows, ncols)
        Frames selected from the input raw data linearly spaced in index of the
        time axis. If n_frames > tot_frames, a number of frames equal to
        tot_frames is returned.
    """
    if isinstance(file_path, str):
        array = h5py_to_numpy(
            file_path, h5py_key, trim_frames_start, trim_frames_end
        )
    elif isinstance(file_path, list):
        array = tiff_to_numpy(file_path, trim_frames_start, trim_frames_end)
    else:
        raise ValueError("File type not supported")
    # Total number of frames in the movie.
    tot_frames = array.shape[0]
    requested_frames = np.linspace(
        0, tot_frames, 1 + min(n_frames, tot_frames), dtype=int
    )[:-1]
    frames = array[requested_frames]
    return frames


def combine_images_with_individual_titles(
    image1_path: Path,
    image2_path: Path,
    output_path: Path,
    title1: str,
    title2: str,
) -> None:
    """Combine two images side-by-side with padding and titles above each image.  # noqa: E501

    Parameters
    ----------
    image1_path : Path
        Path to the first image.
    image2_path : Path
        Path to the second image.
    output_path : Path
        Path to save the combined image.
    title1 : str
         Title text for the first image.
    title2 - str
        Title text for the second image.

    Returns
    -------
    None
    """
    # Open both images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure both images have the same height
    max_height = max(img1.height, img2.height)
    img1 = img1.resize((img1.width, max_height), Image.Resampling.LANCZOS)
    img2 = img2.resize((img2.width, max_height), Image.Resampling.LANCZOS)

    # Set padding and title height
    padding = 20
    title_height = 50  # Space for the titles

    # Calculate dimensions of the combined image
    combined_width = (
        img1.width + img2.width + padding * 3
    )  # Padding between and around images
    combined_height = max_height + padding * 2 + title_height

    # Create a new blank image with padding and room for the titles
    combined_image = Image.new(
        "RGB", (combined_width, combined_height), (255, 255, 255)
    )

    # Draw the titles
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype(
            "arial.ttf", 24
        )  # You can replace with a path to your desired font
    except IOError:
        font = (
            ImageFont.load_default()
        )  # Fallback to default font; may not match expected size

    # Title 1: Above the second image
    bbox1 = draw.textbbox((0, 0), title1, font=font)
    text_width1 = bbox1[2] - bbox1[0]
    text_y1 = padding
    text_x1 = padding + (img1.width - text_width1) // 2
    draw.text((text_x1, text_y1), title1, fill="black", font=font)

    # Title 2: Above the first image
    bbox2 = draw.textbbox((0, 0), title2, font=font)
    text_width2 = bbox2[2] - bbox2[0]
    text_x2 = padding * 2 + img2.width + (img2.width - text_width2) // 2
    text_y2 = padding
    draw.text((text_x2, text_y2), title2, fill="black", font=font)

    # Paste images into the new image
    img1_x = padding
    img2_x = img1_x + img1.width + padding
    img_y = padding + title_height
    combined_image.paste(img1, (img1_x, img_y))
    combined_image.paste(img2, (img2_x, img_y))

    # Save the result
    combined_image.save(output_path)


def compute_residual_optical_flow(
    reg_pc: Union[h5py.Dataset, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the residual optical flow from the registration principal
    components.

    Parameters
    ----------
    reg_pc : Union[h5py.Dataset, np.ndarray]
        Registration principal components.

    Returns
    -------
    residual optical flow : np.ndarray
    average and max shifts : np.ndarray
    """

    regPC = reg_pc
    flows = np.zeros(regPC.shape[1:] + (2,), np.float32)
    for i in range(len(flows)):
        pclow, pchigh = regPC[:, i]
        flows[i] = cv2.calcOpticalFlowFarneback(
            pclow,
            pchigh,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=100,
            iterations=15,
            poly_n=5,
            poly_sigma=1.2 / 5,
            flags=0,
        )
    flows_norm = np.sqrt(np.sum(flows**2, -1))
    farnebackDX = np.transpose(
        [flows_norm.mean((1, 2)), flows_norm.max((1, 2))]
    )
    return flows, farnebackDX


def compute_crispness(
    mov_raw: Union[h5py.Dataset, np.ndarray],
    mov_corr: Union[h5py.Dataset, np.ndarray],
) -> List[float]:
    """Compute the crispness of the raw and corrected movie.

    Parameters
    ----------
    mov_raw : Union[h5py.Dataset, np.ndarray]
        Raw movie data.
    mov_corr : Union[h5py.Dataset, np.ndarray]
        Corrected movie data.

    Returns
    -------
    crispness of mean image : List[float]
    """
    return [
        np.sqrt(np.sum(np.array(np.gradient(mean_image(m))) ** 2))
        for m in (mov_raw, mov_corr)
    ]


def compute_reference(
    input_frames: np.ndarray,
    niter: int,
    maxregshift: float,
    smooth_sigma: float,
    smooth_sigma_time: float,
    mask_slope_factor: float = 3,
) -> np.ndarray:
    """Computes a stacked reference image from the input frames.

    Modified version of Suite2P's compute_reference function with no updating
    of input frames. Picks initial reference then iteratively aligns frames to
    create reference. This code does not reproduce the pre-processing suite2p
    does to data from 1Photon scopes. As such, if processing 1Photon data, the
    user should use the suite2p reference image creation.

    Parameters
    ----------
    input_frames : array-like, (n_frames, nrows, ncols)
        Set of frames to create a reference from.
    niter : int
        Number of iterations to perform when creating the reference image.
    maxregshift : float
        Maximum shift allowed as a fraction of the image width or height, which
        ever is longer.
    smooth_sigma : float
        Width of the Gaussian used to smooth the phase correlation between the
        reference and the frame with which it is being registered.
    smooth_sigma_time : float
        Width of the Gaussian used to smooth between multiple frames by before
        phase correlation.
    mask_slope_factor : int
        Factor to multiply ``smooth_sigma`` by when creating masks for the
        reference image during suite2p phase correlation. These masks down
        weight edges of the image. The default used in suite2p, where this
        method is adapted from, is 3.

    Returns
    -------
    refImg : array-like, (nrows, ncols)
        Reference image created from the input data.
    """
    # Get the dtype of the input frames to properly cast the final reference
    # image as the same type.
    frames_dtype = input_frames.dtype
    # Get initial reference image from suite2p.
    frames = remove_extrema_frames(input_frames)
    ref_image = pick_initial_reference(frames)

    # Determine how much to pad our frames by before shifting to prevent
    # wraps.
    pad_y = int(np.ceil(maxregshift * ref_image.shape[0]))
    pad_x = int(np.ceil(maxregshift * ref_image.shape[1]))

    for idx in range(niter):
        # Compute the number of frames to select in creating the reference
        # image. At most we select half to the input frames.
        nmax = int(frames.shape[0] * (1.0 + idx) / (2 * niter))

        # rigid Suite2P phase registration.
        ymax, xmax, cmax = phasecorr(
            data=apply_masks(
                frames,
                *compute_masks(
                    refImg=ref_image,
                    maskSlope=mask_slope_factor * smooth_sigma,
                ),
            ),
            cfRefImg=phasecorr_reference(
                refImg=ref_image,
                smooth_sigma=smooth_sigma,
            ),
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
        )

        # Find the indexes of the frames that are the most correlated and
        # select the first nmax.
        isort = np.argsort(-cmax)[:nmax]

        # Copy the most correlated frames so we don't shift the original data.
        # We pad this data to prevent wraps from showing up in the reference
        # image. We pad with NaN values to enable us to use nanmean and only
        # average those pixels that contain data in the average.
        max_corr_frames = np.pad(
            array=frames[isort].astype(float),
            pad_width=((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
            constant_values=np.nan,
        )
        max_corr_xmax = xmax[isort]
        max_corr_ymax = ymax[isort]
        # Apply shift to the copy of the frames.
        for frame, dy, dx in zip(
            max_corr_frames, max_corr_ymax, max_corr_xmax
        ):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Create a new reference image from the highest correlated data.
        with warnings.catch_warnings():
            # Assuming the motion correction went well, there should be a lot
            # of empty values in the padded area around the frames. We suppress
            # warnings for these "Empty Slices" as they are expected.
            warnings.filterwarnings("ignore", "Mean of empty slice")
            ref_image = np.nanmean(max_corr_frames, axis=0)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-max_corr_ymax.mean())),
            dx=int(np.round(-max_corr_xmax.mean())),
        )
        # Clip the reference image back down to the original size and remove
        # any NaNs remaining. Throw warning if a NaN is found.
        ref_image = ref_image[pad_y:-pad_y, pad_x:-pad_x]
        if np.any(np.isnan(ref_image)):
            # NaNs can sometimes be left over from the image padding during the
            # first few iterations before the reference image has converged.
            # If there are still NaNs left after the final iteration, we
            # throw the following warning.
            if idx + 1 == niter:
                logging.warning(
                    f"Warning: {np.isnan(ref_image).sum()} NaN pixels were "
                    "found in the reference image on the final iteration. "
                    "Likely the image quality is low and shifting frames "
                    "failed. Setting NaN values to the image mean."
                )
            ref_image = np.nan_to_num(
                ref_image, nan=np.nanmean(ref_image), copy=False
            )
        ref_image = ref_image.astype(frames_dtype)

    return ref_image


def remove_extrema_frames(
    input_frames: np.ndarray, n_sigma: float = 3
) -> np.ndarray:
    """Remove frames with extremum mean values from the frames used in
    reference image processing/creation.

    Likely these are empty frames of pure noise or very high intensity frames
    relative to mean.

    Parameters
    ----------
    input_frames : numpy.ndarray, (N, M, K)
        Set of frames to trim.
    n_sigma : float, optional
        Number of standard deviations to above which to clip. Default is 3
        which was found to remove all empty frames while preserving most
        frames.

    Returns
    -------
    trimmed_frames : numpy.ndarray, (N, M, K)
        Set of frames with the extremum frames removed.
    """
    frame_means = np.mean(input_frames, axis=(1, 2))
    _, low_cut, high_cut = sigmaclip(frame_means, low=n_sigma, high=n_sigma)
    trimmed_frames = input_frames[
        np.logical_and(frame_means > low_cut, frame_means < high_cut)
    ]
    return trimmed_frames


def optimize_motion_parameters(
    initial_frames: np.ndarray,
    smooth_sigmas: np.array,
    smooth_sigma_times: np.array,
    suite2p_args: dict,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    logger: Optional[Callable] = None,
) -> dict:
    """Loop over a range of parameters and select the best set from the
    max acutance of the final, average image.

    Parameters
    ----------
    initial_frames : numpy.ndarray, (N, M, K)
        Smaller subset of frames to create a reference image from.
    smooth_sigmas : numpy.ndarray, (N,)
        Array of suite2p smooth sigma values to attempt. Number of iterations
        will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    smooth_sigma_times : numpy.ndarray, (N,)
        Array of suite2p smooth sigma time values to attempt. Number of
        iterations will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    suite2p_args : dict
        A dictionary of suite2p configs containing at minimum:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Processing a large number of frames at once
        will likely result in running out of memory, hence processing in
        batches. Total returned size isn_batches * suit2p_args['batch_size'].
    logger : Optional[Callable]
        Function to print to stdout or a log.

    Returns
    -------
    best_result : dict
        A dict containing the final results of the search:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``ref_image``
            Reference Image created with the settings yielding the highest
            image acutance (numpy.ndarray, (N, M))
        ``acutance``
            Acutance of ``best_image``. (float)
        ``smooth_sigma``
            Value of ``smooth_sigma`` found to yield the best acutance (float).
        ``smooth_sigma_time``
            Value of ``smooth_sigma_time`` found to yield the best acutance
            (float).
    """
    best_results = {
        "acutance": 1e-16,
        "ave_image": np.array([]),
        "ref_image": np.array([]),
        "smooth_sigma": -1,
        "smooth_sigma_time": -1,
    }
    logger("Starting search for best smoothing parameters...")
    sub_frames = load_representative_sub_frames(
        suite2p_args["h5py"],
        suite2p_args["h5py_key"],
        trim_frames_start,
        trim_frames_end,
        n_batches=n_batches,
        batch_size=suite2p_args["batch_size"],
    )
    start_time = time()
    for param_spatial, param_time in product(
        smooth_sigmas, smooth_sigma_times
    ):
        current_args = suite2p_args.copy()
        current_args["smooth_sigma"] = param_spatial
        current_args["smooth_sigma_time"] = param_time

        if logger:
            logger(
                f'\tTrying: smooth_sigma={current_args["smooth_sigma"]}, '
                f'smooth_sigma_time={current_args["smooth_sigma_time"]}'
            )

        ref_image = compute_reference(
            initial_frames,
            8,
            current_args["maxregshift"],
            current_args["smooth_sigma"],
            current_args["smooth_sigma_time"],
        )
        image_results = create_ave_image(
            ref_image,
            sub_frames.copy(),
            current_args,
            batch_size=suite2p_args["batch_size"],
        )
        ave_image = image_results["ave_image"]
        # Compute the acutance ignoring the motion border. Sharp motion
        # borders can potentially get rewarded with high acutance.
        current_acu = compute_acutance(
            ave_image,
            image_results["min_y"],
            image_results["max_y"],
            image_results["min_x"],
            image_results["max_x"],
        )

        if current_acu > best_results["acutance"]:
            best_results["acutance"] = current_acu
            best_results["ave_image"] = ave_image
            best_results["ref_image"] = ref_image
            best_results["smooth_sigma"] = current_args["smooth_sigma"]
            best_results["smooth_sigma_time"] = current_args[
                "smooth_sigma_time"
            ]
        if logger:
            logger(f"\t\tResulting acutance={current_acu:.4f}")
    if logger:
        logger(
            f"Found best motion parameters in {time() - start_time:.0f} "
            f'seconds, with image acutance={best_results["acutance"]:.4f}, '
            f'for parameters: smooth_sigma={best_results["smooth_sigma"]}, '
            f'smooth_sigma_time={best_results["smooth_sigma_time"]}'
        )
    return best_results


def load_representative_sub_frames(
    h5py_name,
    h5py_key,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    batch_size: int = 500,
):
    """Load a subset of frames spanning the full movie.

    Parameters
    ----------
    h5py_name : str
        Path to the h5 file to load frames from.
    h5py_key : str
        Name of the h5 dataset containing the movie.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Total returned size is
        n_batches * batch_size.
    batch_size : int, optional
        Number of frames to process at once. Total returned size is
        n_batches * batch_size.

    Returns
    -------
    """
    output_frames = []
    frame_fracts = np.arange(0, 1, 1 / n_batches)
    with h5py.File(h5py_name, "r") as h5_file:
        dataset = h5_file[h5py_key]
        total_frames = dataset.shape[0] - trim_frames_start - trim_frames_end
        if total_frames < n_batches * batch_size:
            return dataset[:]
        for percent_start in frame_fracts:
            frame_start = int(percent_start * total_frames + trim_frames_start)
            output_frames.append(
                dataset[frame_start : frame_start + batch_size]
            )
    return np.concatenate(output_frames)


def create_ave_image(
    ref_image: np.ndarray,
    input_frames: np.ndarray,
    suite2p_args: dict,
    batch_size: int = 500,
) -> dict:
    """Run suite2p image motion correction over a full movie.

    Parameters
    ----------
    ref_image : numpy.ndarray, (N, M)
        Reference image to correlate with movie frames.
    input_frames : numpy.ndarray, (L, N, M)
        Frames to motion correct and compute average image/acutance of.
    suite2p_args : dict
        Dictionary of suite2p args containing:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
        ``"smooth_sigma"``
            Spatial Gaussian smoothing parameter used by suite2p to smooth
            frames before correlation. (float).
        ``"smooth_sigma_time"``
            Time Gaussian smoothing of frames to apply before correlation.
            (float).
    batch_size : int, optional
        Number of frames to process at once.

    Returns
    -------
    ave_image_dict : dict
        A dict containing the average image and motion border values:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``min_y``
            Minimum y allowed value in image array. Below this is motion
            border.
        ``max_y``
            Maximum y allowed value in image array. Above this is motion
            border.
        ``min_x``
            Minimum x allowed value in image array. Below this is motion
            border.
        ``max_x``
            Maximum x allowed value in image array. Above this is motion
            border.
    """
    ave_frame = np.zeros((ref_image.shape[0], ref_image.shape[1]))
    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0
    tot_frames = input_frames.shape[0]
    add_modify_required_parameters(suite2p_args)
    for start_idx in np.arange(0, tot_frames, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > tot_frames:
            end_idx = tot_frames
        frames = input_frames[start_idx:end_idx]
        frames, dy, dx, _, _, _, _ = register_frames(
            refAndMasks=ref_image, frames=frames, ops=suite2p_args
        )
        min_y = min(min_y, dy.min())
        max_y = max(max_y, dy.max())
        min_x = min(min_x, dx.min())
        max_x = max(max_x, dx.max())
        ave_frame += frames.sum(axis=0) / tot_frames

    return {
        "ave_image": ave_frame,
        "min_y": int(np.fabs(min_y)),
        "max_y": int(max_y),
        "min_x": int(np.fabs(min_x)),
        "max_x": int(max_x),
    }


def add_modify_required_parameters(suite2p_args: dict):
    """Check that minimum parameters needed by suite2p registration are
    available. If not add them to the suite2p_args dict.

    Additionally, make sure that nonrigid is set to false as are gridsearch
    of parameters above is not setup to use nonrigid.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary with potentially missing values.
    """
    if suite2p_args.get("1Preg") is None:
        suite2p_args["1Preg"] = False
    if suite2p_args.get("bidiphase") is None:
        suite2p_args["bidiphase"] = False
    if suite2p_args.get("nonrigid") is None:
        suite2p_args["nonrigid"] = False
    if suite2p_args.get("norm_frames") is None:
        suite2p_args["norm_frames"] = True
    # Don't use nonrigid for parameter search.
    suite2p_args["nonrigid"] = False


def compute_acutance(
    image: np.ndarray,
    min_cut_y: int = 0,
    max_cut_y: int = 0,
    min_cut_x: int = 0,
    max_cut_x: int = 0,
) -> float:
    """Compute the acutance (sharpness) of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute acutance of.
    min_cut_y : int
        Number of pixels to cut from the beginning of the y axis.
    max_cut_y : int
        Number of pixels to cut from the end of the y axis.
    min_cut_x : int
        Number of pixels to cut from the beginning of the x axis.
    max_cut_x : int
        Number of pixels to cut from the end of the x axis.

    Returns
    -------
    acutance : float
        Acutance of the image.
    """
    im_max_y, im_max_x = image.shape

    cut_image = image[
        min_cut_y : im_max_y - max_cut_y, min_cut_x : im_max_x - max_cut_x
    ]
    grady, gradx = np.gradient(cut_image)
    return (grady**2 + gradx**2).mean()


def check_and_warn_on_datatype(
    filepath: Union[Path, list],
    logger: Callable,
    filetype: str = "h5",
    h5py_key: str = "",
):
    """Suite2p assumes int16 types throughout code. Check that the input
    data is type int16 else throw a warning.

    Parameters
    ----------
    filepath : Union[Path, list]
        Path to the HDF5 containing the data.
    logger : Callable
        Logger to output logger warning to.
    filetype : str
        Type of file to check. Default is "h5".
    h5py_key : str
        Name of the dataset to check. Default is "".

    """
    if filetype == "h5":
        byteorder, name = h5py_byteorder_name(filepath, h5py_key)
    elif filetype == "tiff":
        byteorder, name = tiff_byteorder_name(filepath)
    else:
        raise ValueError("File type not supported")
    if byteorder == ">":
        logger(
            "Data byteorder is big-endian which may cause issues in "
            "suite2p. This may result in a crash or unexpected "
            "results."
        )
    if name != "int16":
        logger(
            f"Data type is {name} and not int16. Suite2p "
            "assumes int16 data as input and throughout codebase. "
            "Non-int16 data may result in unexpected results or "
            "crashes."
        )


def _mean_of_batch(i, array):
    """Return the per-frame mean of a 1000-frame batch starting at i."""
    return array[i : i + 1000].mean(axis=(1, 2))


def find_movie_start_end_empty_frames(
    filepath: Union[str, list[str]],
    h5py_key: str = "",
    n_sigma: float = 5,
    logger: Optional[Callable] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[int, int]:
    """Load a movie from HDF5 and find frames at the start and end of the
    movie that are empty or pure noise and 5 sigma discrepant from the
    average frame.

    If a non-contiguous set of frames is found, the code will return 0 for
    that half of the movie and throw a warning about the quality of the data.

    Parameters
    ----------
    h5py_name : str
        Name of the HDF5 file to load from.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_sigma : float
        Number of standard deviations beyond which a frame is considered an
        outlier and "empty".
    logger : Optional[Callable]
        Function to print warning messages to.
    n_jobs: Optional[int]
        The number of jobs to run in parallel.

    Returns
    -------
    trim_frames : Tuple[int, int]
        Tuple of the number of frames to cut from the start and end of the
        movie as (n_trim_start, n_trim_end).
    """
    if isinstance(filepath, str):
        array = h5py_to_numpy(filepath, h5py_key)
    else:
        array = tiff_to_numpy(filepath)
    # Find the midpoint of the movie.
    n_frames = array.shape[0]
    midpoint = n_frames // 2
    # We discover empty or extrema frames by comparing the mean of each frames
    # to the mean of the full movie.
    if n_jobs == 1 or n_frames < 2000:
        means = array[:].mean(axis=(1, 2))
    else:
        means = np.concatenate(
            ThreadPool(n_jobs).starmap(
                _mean_of_batch,
                product(range(0, n_frames, 1000), [array]),
            )
        )
    mean_of_frames = means.mean()

    # Compute a robust standard deviation that is not sensitive to the
    # outliers we are attempting to find.
    quart_low, quart_high = np.percentile(means, [25, 75])
    # Convert the inner quartile range to an estimate of the standard deviation
    # 0.6745 is the converting factor between the inner quartile and a
    # traditional standard deviation.
    std_est = (quart_high - quart_low) / (2 * 0.6745)

    # Get the indexes of the frames that are found to be n_sigma deviating.
    start_idxs = np.sort(
        np.argwhere(means[:midpoint] < mean_of_frames - n_sigma * std_est)
    ).flatten()
    end_idxs = (
        np.sort(
            np.argwhere(means[midpoint:] < mean_of_frames - n_sigma * std_est)
        ).flatten()
        + midpoint
    )

    # Get the total number of these frames.
    lowside = len(start_idxs)
    highside = len(end_idxs)

    # Check to make sure that the indexes found were only from the start/end
    # of the movie. If not, throw a warning and reset the number of frames
    # found to zero.
    if not np.array_equal(
        start_idxs, np.arange(0, lowside, dtype=start_idxs.dtype)
    ):
        lowside = 0
        if logger is not None:
            logger(
                f"{n_sigma} sigma discrepant frames found outside the "
                "beginning of the movie. Please inspect the movie for data "
                "quality. Not trimming frames from the movie beginning."
            )
    if not np.array_equal(
        end_idxs,
        np.arange(n_frames - highside, n_frames, dtype=end_idxs.dtype),
    ):
        highside = 0
        if logger is not None:
            logger(
                f"{n_sigma} sigma discrepant frames found outside the end "
                "of the movie. Please inspect the movie for data quality. "
                "Not trimming frames from the movie end."
            )

    return (lowside, highside)


def reset_frame_shift(
    frames: np.ndarray,
    dy_array: np.ndarray,
    dx_array: np.ndarray,
    trim_frames_start: int,
    trim_frames_end: int,
):
    """Reset the frames of a movie and their shifts.

    Shifts the frame back to its original location and resets the shifts for
    those frames to (0, 0). Frames, dy_array, and dx_array are edited in
    place.

    Parameters
    ----------
    frames : numpy.ndarray, (N, M, K)
        Full movie to reset frames in.
    dy_array : numpy.ndarray, (N,)
        Array of shifts in the y direction for each frame of the movie.
    dx_array : numpy.ndarray, (N,)
        Array of shifts in the x direction for each frame of the movie.
    trim_frames_start : int
        Number of frames at the start of the movie that were identified as
        empty or pure noise.
    trim_frames_end : int
        Number of frames at the end of the movie that were identified as
        empty or pure noise.
    """
    for idx in range(trim_frames_start):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0

    for idx in range(frames.shape[0] - trim_frames_end, frames.shape[0]):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0


def projection_process(
    data: np.ndarray, projection: str = "max"
) -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError(
            'projection can be "max" or "avg" not ' f"{projection}"
        )
    return normalize_array(proj)


def identify_and_clip_outliers(
    data: np.ndarray, med_filter_size: int, thresh: int
) -> Tuple[np.ndarray, np.ndarray]:
    """given data, identify the indices of outliers
    based on median filter detrending, and a threshold

    Parameters
    ----------
    data: np.ndarray
        1D array of samples
    med_filter_size: int
        the number of samples for 'size' in
        scipy.ndimage.filters.median_filter
    thresh: int
        multipled by the noise estimate to establish a threshold, above
        which, samples will be marked as outliers.

    Returns
    -------
    data: np.ndarry
        1D array of samples, clipped to threshold around median-filtered data
    indices: np.ndarray
        the indices where clipping took place

    """
    data_filtered = median_filter(data, med_filter_size, mode="nearest")
    detrended = data - data_filtered
    indices = np.argwhere(np.abs(detrended) > thresh).flatten()
    data[indices] = np.clip(
        data[indices],
        data_filtered[indices] - thresh,
        data_filtered[indices] + thresh,
    )
    return data, indices


def make_output_directory(output_dir: Path, plane: str = "") -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    plane: str
        plane number

    Returns
    -------
    output_dir: Path
        output directory
    """
    if not plane:
        output_dir = output_dir / "motion_correction"
    else:
        output_dir = output_dir / plane / "motion_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def check_trim_frames(data):
    """Make sure that if the user sets auto_remove_empty_frames
    and timing frames is already requested, raise an error.
    """
    if data["auto_remove_empty_frames"] and (
        data["trim_frames_start"] > 0 or data["trim_frames_end"] > 0
    ):
        msg = (
            "Requested auto_remove_empty_frames but "
            "trim_frames_start > 0 or trim_frames_end > 0. Please "
            "either request auto_remove_empty_frames or manually set "
            "trim_frames_start/trim_frames_end if number of frames to "
            "trim is known."
        )
        raise ValueError(msg)
    return data


def make_png(
    max_proj_path: Path,
    avg_proj_path: Path,
    summary_df: pd.DataFrame,
    dst_path: Path,
):
    """Write the registration-summary PNG (projections + xy offsets)."""
    xo = np.abs(summary_df["x"]).max()
    yo = np.abs(summary_df["y"]).max()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 4)
    mx_ax = fig.add_subplot(gs[0:2, 0:2])
    av_ax = fig.add_subplot(gs[0:2, 2:4])
    xyax = fig.add_subplot(gs[2, :])
    corrax = fig.add_subplot(gs[3, :])

    for ax, im_path in zip([mx_ax, av_ax], [max_proj_path, avg_proj_path]):
        with Image.open(im_path) as im:
            ax.imshow(im, cmap="gray")
            sz = im.size
        ax.axvline(xo, color="r", linestyle="--")
        ax.axvline(sz[0] - xo, color="r", linestyle="--")
        ax.axhline(yo, color="g", linestyle="--")
        ax.axhline(sz[1] - yo, color="g", linestyle="--")
        ax.set_title(f"{im_path.parent}\n{im_path.name}", fontsize=8)

    xyax.plot(summary_df["x"], linewidth=0.5, color="r", label="xoff")
    xyax.axhline(xo, color="r", linestyle="--")
    xyax.axhline(-xo, color="r", linestyle="--")
    xyax.plot(
        summary_df["y"], color="g", linewidth=0.5, alpha=0.5, label="yoff"
    )
    xyax.axhline(yo, color="g", linestyle="--")
    xyax.axhline(-yo, color="g", linestyle="--")
    xyax.legend(loc=0)
    xyax.set_ylabel("correction offset [pixels]")

    corrax.plot(
        summary_df["correlation"], color="k", linewidth=0.5, label="corrXY"
    )
    corrax.set_xlabel("frame index")
    corrax.set_ylabel("correlation peak value")
    corrax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(dst_path)

    return dst_path


def make_nonrigid_png(
    output_path: Path,
    avg_proj_path: Path,
    summary_df: pd.DataFrame,
    dst_path: Path,
):
    """Write the per-block nonrigid registration-summary PNG."""
    nonrigid_y = np.array(
        list(map(eval, summary_df["nonrigid_y"])), dtype=np.float32
    )
    nonrigid_x = np.array(
        list(map(eval, summary_df["nonrigid_x"])), dtype=np.float32
    )
    nonrigid_corr = np.array(
        list(map(eval, summary_df["nonrigid_corr"])), dtype=np.float32
    )
    ops = json.loads(h5py.File(output_path)["metadata"][()].decode())[
        "suite2p_args"
    ]
    with Image.open(avg_proj_path) as im:
        Ly, Lx = im.size
    yblock, xblock = make_blocks(Ly=Ly, Lx=Lx, block_size=ops["block_size"])[
        :2
    ]
    nblocks = len(xblock)

    fig = plt.figure(figsize=(22, 3 * nblocks))
    gs = fig.add_gridspec(25 * nblocks, 6)
    for i in range(nblocks):
        av_ax = fig.add_subplot(gs[25 * i : 25 * i + 20, 0])
        xyax = fig.add_subplot(gs[25 * i : 25 * i + 10, 1:])
        corrax = fig.add_subplot(gs[25 * i + 10 : 25 * i + 20, 1:])

        with Image.open(avg_proj_path) as im:
            av_ax.imshow(im, cmap="gray")
            sz = im.size
            av_ax.set_ylim(0, sz[0])
            av_ax.set_xlim(0, sz[1])
        for x in xblock[i]:
            av_ax.vlines(x, *yblock[i], color="r", linestyle="--")
        for y in yblock[i]:
            av_ax.hlines(y, *xblock[i], color="g", linestyle="--")

        xyax.plot(nonrigid_x[:, i], linewidth=0.5, color="r", label="xoff")
        xyax.plot(
            nonrigid_y[:, i], color="g", linewidth=0.5, alpha=0.5, label="yoff"
        )
        if i == 0:
            xyax.legend(loc=0)
        xyax.set_xticks([])
        xyax.set_xlim(0, nonrigid_x.shape[0])
        xyax.set_ylabel("offset [pixels]")

        corrax.plot(
            nonrigid_corr[:, i], color="k", linewidth=0.5, label="corrXY"
        )
        corrax.set_xlim(0, nonrigid_x.shape[0])
        corrax.set_xlabel("frame index")
        corrax.set_ylabel("correlation")
        if i == 0:
            corrax.legend(loc=0)
    fig.savefig(dst_path, bbox_inches="tight")

    return dst_path


def downsample_normalize(
    movie_path: Path,
    frame_rate: float,
    bin_size: float,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    """reads in a movie (nframes x nrows x ncols), downsamples,
    creates an average projection, and normalizes according to
    quantiles in that projection.

    Parameters
    ----------
    movie_path: Path
        path to an h5 file, containing an (nframes x nrows x ncol) dataset
        named 'data'
    frame_rate: float
        frame rate of the movie specified by 'movie_path'
    bin_size: float
        desired duration in seconds of a downsampled bin, i.e. the reciprocal
        of the desired downsampled frame rate.
    lower_quantile: float
        arg supplied to `np.quantile()` to determine lower cutoff value from
        avg projection for normalization.
    upper_quantile: float
        arg supplied to `np.quantile()` to determine upper cutoff value from
        avg projection for normalization.

    Returns
    -------
    ds: np.ndarray
        a downsampled and normalized array

    Notes
    -----
    This strategy was satisfactory in the labeling app for maintaining
    consistent visibility.

    """
    if isinstance(movie_path, Path):
        ds = downsample_h5_video(
            movie_path, input_fps=frame_rate, output_fps=1.0 / bin_size
        )
    else:
        ds = downsample_array(
            movie_path, input_fps=frame_rate, output_fps=1.0 / bin_size
        )
    avg_projection = ds.mean(axis=0)
    lower_cutoff, upper_cutoff = np.quantile(
        avg_projection.flatten(), (lower_quantile, upper_quantile)
    )
    ds = normalize_array(
        ds, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff
    )
    return ds


def flow_png(output_path: Path, dst_path: str, iPC: int = 0):
    """Write PC_low/PC_high/residual-optical-flow images for one PC."""
    with h5py.File(output_path) as f:
        regPC = f["reg_metrics/regPC"]
        tPC = f["reg_metrics/tPC"]
        flows = f["reg_metrics/farnebackROF"]
        flow_ds = np.array(
            [
                cv2.resize(flows[iPC, :, :, a], dsize=None, fx=0.1, fy=0.1)
                for a in (0, 1)
            ]
        )
        flow_ds_norm = np.sqrt(np.sum(flow_ds**2, 0))
        # redo Suite2p's PCA-based frame selection
        n_frames, Ly, Lx = f["data"].shape
        nsamp = min(
            2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames
        )
        inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
        nlowhigh = np.minimum(300, int(n_frames / 2))
        isort = np.argsort(tPC, axis=0)

        for k in (0, 1):
            f, a = plt.subplots(2, figsize=(5, 6))
            a[0].set_position([0.08, 0.92, 0.88, 0.08])
            a[0].hist(
                np.sort(
                    inds[
                        isort[-nlowhigh:, iPC] if k else isort[:nlowhigh, iPC]
                    ]
                ),
                50,
            )
            a[0].set_title(
                "averaged frames for " + ("$PC_{high}$" if k else "$PC_{low}$")
            )
            a[1].set_position([0, 0, 1, 0.9])
            vmin = np.min(regPC[1 if k else 0, iPC])
            vmax = 5 * np.median(regPC[1 if k else 0, iPC]) - 4 * vmin
            a[1].imshow(
                regPC[1 if k else 0, iPC], cmap="gray", vmin=vmin, vmax=vmax
            )
            a[1].axis("off")
            plt.savefig(
                dst_path + (f"_PC{iPC}low.png", f"_PC{iPC}high.png")[k],
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
        f, a = plt.subplots(2, figsize=(5, 6))
        a[0].set_position([0.06, 0.95, 0.9, 0.05])
        a[1].set_position([0, 0, 1, 0.9])
        im = a[1].quiver(
            *flow_ds[:, ::-1], flow_ds_norm[::-1]
        )  # imshow puts origin [0,0] in upper left
        a[1].axis("off")
        plt.colorbar(im, cax=a[0], location="bottom")
        a[0].set_title("residual optical flow")
        plt.savefig(
            dst_path + f"_PC{iPC}rof.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
        )


def get_frame_rate_from_sync(sync_file, platform_data) -> float:
    """Calculate frame rate from sync file
    Parameters
    ----------
    sync_file: str
        path to sync file
    platform_data: dict
        platform data from platform.json
    Returns
    -------
    frame_rate_hz: float
        frame rate in Hz
    """
    labels = [
        "vsync_2p",
        "2p_vsync",
    ]  # older versions of sync may 2p_vsync label
    imaging_groups = len(
        platform_data["imaging_plane_groups"]
    )  # Number of imaging plane groups for frequency calculation
    frame_rate_hz = None
    for i in labels:
        logging.info(f"Pulling framerate from sync file: {sync_file}")
        sync_data = Sync(sync_file)
        try:
            rising_edges = sync_data.get_rising_edges(i, units="seconds")
            image_freq = 1 / (np.mean(np.diff(rising_edges)))
            frame_rate_hz = image_freq / imaging_groups

        except ValueError:
            pass
    sync_data.close()
    if not frame_rate_hz:
        raise ValueError(
            f"Frame rate no acquired, line labels: {sync_data.line_labels}"
        )
    return frame_rate_hz


def multiplane_motion_correction(
    data_dir: Path, output_dir: Path, debug: bool = False
):
    """Process multiplane data for suite2p parameters

    Parameters
    ----------
    data_dir: Path
        path to h5 file
    output_dir: Path
        output directory
    debug: bool
        run in debug mode
    Returns
    -------
    h5_file: Path
        path to h5 file
    output_dir: Path
        output directory
    frame_rate_hz: float
        frame rate in Hz
    """
    pattern = re.compile(r"^V.*\d+$")
    matching_files = [
        d for d in data_dir.rglob("*.txt") if pattern.match(d.stem)
    ]
    if len(matching_files) > 0:
        with open(matching_files[0], "r") as f:
            h5_file = f.read()
        h5_file = next(data_dir.rglob(h5_file))
        unique_id = h5_file.stem
    else:
        h5_dir = [i for i in data_dir.rglob("*V*") if i.is_dir()][0]
        unique_id = h5_dir.name
        h5_file = [i for i in h5_dir.glob(f"{h5_dir.name}.h5")][0]
    logging.info("Found raw time series to process %s", h5_file)
    session_fp = next(data_dir.rglob("session.json"), "")
    if not session_fp:
        raise f"Could not locate session.json in {session_fp}"
    with open(session_fp) as f:
        session_data = json.load(f)
    output_dir = make_output_directory(output_dir, unique_id)
    try:
        frame_rate_hz = float(
            session_data["data_streams"][0]["ophys_fovs"][0]["frame_rate"]
        )
    except KeyError:
        logging.warning(
            "Frame rate not found in session.json, pulling from platform.json"
        )
        platform_json = next(data_dir.rglob("*platform.json"))
        with open(platform_json, "r") as j:
            platform_data = json.load(j)
        sync_file = [i for i in data_dir.rglob("platform_data['sync_file']")][
            0
        ]
        frame_rate_hz = get_frame_rate_from_sync(sync_file, platform_data)
    if debug:
        logging.info("Running in debug mode....")
        raw_data = h5py.File(h5_file, "r")
        frames_6min = int(360 * float(frame_rate_hz))
        trimmed_data = raw_data["data"][:frames_6min]
        raw_data.close()
        trimmed_fn = Path("../scratch") / f"{unique_id}.h5"
        with h5py.File(trimmed_fn, "w") as f:
            f.create_dataset("data", data=trimmed_data)
        h5_file = trimmed_fn
    return h5_file, output_dir, frame_rate_hz


def update_suite2p_args_reference_image(
    suite2p_args: dict, args: dict, reference_image_fp=None
):
    """Update the suite2p_args dictionary with the reference image.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary.
    args : dict
        Dictionary of arguments from the command line.
    reference_image_fp : Path
        Path to the reference image to use. Default is None.

    Returns
    -------
    suite2p_args : dict
        Updated suite2p_args dictionary.
    args : dict
        Updated args dictionary.
    """
    # Use our own version of compute_reference to create the initial
    # reference image used by suite2p.
    logger.info(
        f'Loading {suite2p_args["nimg_init"]} frames '
        "for reference image creation."
    )
    if reference_image_fp:
        initial_frames = load_initial_frames(
            file_path=reference_image_fp,
            h5py_key=suite2p_args["h5py_key"],
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
        )

    else:
        if suite2p_args.get("h5py", None):
            file_path = suite2p_args["h5py"]
            h5py_key = suite2p_args["h5py_key"]
        else:
            file_path = suite2p_args["tiff_list"]
            h5py_key = None
        initial_frames = load_initial_frames(
            file_path=file_path,
            h5py_key=h5py_key,
            n_frames=suite2p_args["nimg_init"],
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
        )

    if args["do_optimize_motion_params"]:
        logger.info("Attempting to optimize registration parameters Using:")
        logger.info(
            "\tsmooth_sigma range: "
            f'{args["smooth_sigma_min"]} - '
            f'{args["smooth_sigma_max"]}, '
            f'steps: {args["smooth_sigma_steps"]}'
        )
        logger.info(
            "\tsmooth_sigma_time range: "
            f'{args["smooth_sigma_time_min"]} - '
            f'{args["smooth_sigma_time_max"]}, '
            f'steps: {args["smooth_sigma_time_steps"]}'
        )

        # Create linear spaced arrays for the range of smooth
        # parameters to try.
        smooth_sigmas = np.linspace(
            args["smooth_sigma_min"],
            args["smooth_sigma_max"],
            args["smooth_sigma_steps"],
        )
        smooth_sigma_times = np.linspace(
            args["smooth_sigma_time_min"],
            args["smooth_sigma_time_max"],
            args["smooth_sigma_time_steps"],
        )

        optimize_result = optimize_motion_parameters(
            initial_frames=initial_frames,
            smooth_sigmas=smooth_sigmas,
            smooth_sigma_times=smooth_sigma_times,
            suite2p_args=suite2p_args,
            trim_frames_start=args["trim_frames_start"],
            trim_frames_end=args["trim_frames_end"],
            n_batches=args["n_batches"],
            logger=logger.info,
        )
        if args["use_ave_image_as_reference"]:
            suite2p_args["refImg"] = optimize_result["ave_image"]
        else:
            suite2p_args["refImg"] = optimize_result["ref_image"]
        suite2p_args["smooth_sigma"] = optimize_result["smooth_sigma"]
        suite2p_args["smooth_sigma_time"] = optimize_result[
            "smooth_sigma_time"
        ]
    else:
        # Create the initial reference image and store it in the
        # suite2p_args dictionary. 8 iterations is the current default
        # in suite2p.
        tic = -time()
        logger.info("Creating custom reference image...")
        suite2p_args["refImg"] = compute_reference(
            input_frames=initial_frames,
            niter=args["max_reference_iterations"],
            maxregshift=suite2p_args["maxregshift"],
            smooth_sigma=suite2p_args["smooth_sigma"],
            smooth_sigma_time=suite2p_args["smooth_sigma_time"],
        )
        tic += time()
        logger.info(f"took {tic}s")
    return suite2p_args, args


def generate_single_plane_reference(fp: Path, session) -> Path:
    """Generate virtual movies for Bergamo data

    Parameters
    ----------
    fp: Path
        path to h5 file
    session: dict
        session metadata
    Returns
    -------
    Path
        path to reference image
    """
    with h5py.File(fp, "r") as f:
        # take the first bci epoch to save out reference image TODO
        tiff_stems = json.loads(f["epoch_locations"][:][0])
        bci_epochs = [
            i
            for i in session["stimulus_epochs"]
            if i["stimulus_name"] == "single neuron BCI conditioning"
        ]
        bci_epoch_loc = [
            i["output_parameters"]["tiff_stem"] for i in bci_epochs
        ][0]
        frame_length = (
            tiff_stems[bci_epoch_loc][1] - tiff_stems[bci_epoch_loc][0]
        )
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(
            shape=(frame_length, *f["data"].shape[1:]), dtype=f["data"].dtype
        )
        layout[0:frame_length] = vsource[
            tiff_stems[bci_epoch_loc][0] : tiff_stems[bci_epoch_loc][1]
        ]

        with h5py.File("../scratch/reference_image.h5", "w") as ref:

            ref.create_virtual_dataset("data", layout)
    return Path("../scratch/reference_image.h5")


def singleplane_motion_correction(
    h5_file: Path,
    output_dir: Path,
    session,
    unique_id: str,
    debug: bool = False,
):
    """Process single plane data for suite2p parameters

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    output_dir: Path
        output directory
    session: dict
        session metadata
    unique_id: str
        experiment id from data description
    debug: bool

    Returns
    -------
    h5_file: str
        path to h5 file
    output_dir: Path
        output directory
    reference_image_fp: str
        path to reference image
    """
    if not h5_file.is_file():
        h5_file = [f for f in h5_file.rglob("*.h5") if unique_id in str(f)][0]
    output_dir = make_output_directory(output_dir, unique_id)
    reference_image_fp = generate_single_plane_reference(h5_file, session)
    if debug:
        stem = h5_file.stem
        debug_file = Path("../scratch") / f"{stem}_debug.h5"
        with h5py.File(h5_file, "r") as f:
            data = f["data"][:5000]
            trial_locations = f["trial_locations"][()]
            epoch_filenames = f["epoch_locations"][()]
        with h5py.File(debug_file, "a") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("trial_locations", data=trial_locations)
            f.create_dataset("epoch_locations", data=epoch_filenames)
        h5_file = debug_file
    with h5py.File(h5_file, "r") as f:
        trial_locations = json.loads(f["trial_locations"][:][0])
        epoch_locations = json.loads(f["epoch_locations"][:][0])
    with open(output_dir / "trial_locations.json", "w") as j:
        json.dump(trial_locations, j)
    with open(output_dir / "epoch_locations.json", "w") as j:
        json.dump(epoch_locations, j)

    return str(h5_file), output_dir, str(reference_image_fp)

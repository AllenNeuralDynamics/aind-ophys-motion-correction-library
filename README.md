# aind-ophys-motion-correction-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.10,<3.13-blue?logo=python)

Suite2P-based motion correction for AIND optical physiology, with
`aind-data-schema` v2 metadata I/O.

This is the backing library for the `aind-ophys-motion-correction` Code Ocean
capsule. The capsule is a thin wrapper that calls `job.run()`; all logic,
tests and dependency pins live here.

## Usage

```python
from aind_ophys_motion_correction_library.job import run

run()  # settings come from CLI args / environment
```

`run()` reads the acquisition metadata (schema v1 `session.json` or v2
`acquisition.json`), resolves which plane this task holds, registers the movie
with Suite2P, and writes the outputs plus a full `processing.json` and
`quality_control.json`.

## Modules

### `job.py`

Orchestration. `run()` is the entry point: metadata read, logging setup, plane
identity, Suite2P registration, output writing, and the `processing.json` /
`quality_control.json` emission (via the shared `stage_guard`).

### `registration.py`

Suite2P algorithm and per-plane data preparation: reference-image creation
(`compute_reference`), empty-frame detection
(`find_movie_start_end_empty_frames`), outlier clipping
(`identify_and_clip_outliers`), projections (`projection_process`), summary
plots (`make_png`, `make_nonrigid_png`, `flow_png`), registration metrics
(`compute_crispness`, `compute_residual_optical_flow`), and the input
discovery helpers (`find_multiplane_input`, `find_singleplane_input`,
`singleplane_motion_correction`, `truncate_for_debug`).

### `settings.py`

`MotionCorrectionSettings` — the pydantic-settings model behind the CLI, the
environment (`MOTION_CORRECTION_` prefix) and the Code Ocean app panel.

### `qc.py`

Motion-correction QC metric content (`build_registration_summary_metric`,
`build_fov_quality_metric`) over the shared `aind_pophys_metadata` builders.
One `quality_control.json` is written per plane.

### `sync_dataset.py`

Vendored `Sync` reader for sync `.h5` files, used only as a last-resort frame
rate source.

### `utils/metadata_utils.py`

The `MotionCorrectionMetadata` view over v1/v2 metadata
(`load_motion_correction_metadata`), canonical plane-id resolution
(`resolve_plane_id`), the Bergamo conditioning-epoch TIFF stem
(`bci_epoch_tiff_stem`), and the `Code` block (`build_code`).

## Outputs

Written into `<results>/<plane_id>/motion_correction/`, each named after the
plane basename:

| File | Contents |
|---|---|
| `*_registered.h5` | registered movie, `ref_image`, run `metadata`, `reg_metrics` |
| `*_motion_transform.csv` | per-frame x/y offsets, correlation, validity, and per-block columns when nonrigid |
| `*_maximum_projection.png` | max projection |
| `*_average_projection.png` | average projection |
| `*_registration_summary.png` | projections plus offset/correlation traces (`*_nonrigid` variant when nonrigid) |
| `*_motion_preview.webm` | raw-vs-registered preview movie |
| `processing.json` | full `Processing` document for this plane |
| `quality_control.json` | full `QualityControl` document for this plane |

Bergamo single-plane runs additionally write `trial_locations.json` and
`epoch_locations.json`. `flow_png` writes `*_PC<n>low/high/rof.png` beside the
registration summary when registration principal components are available.

## Installation

To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

## Tests and linters

Tests are `unittest`. The CI gate is:
```bash
uv run flake8 . && uv run interrogate --verbose .
uv run coverage run -m unittest discover && uv run coverage report
```
Coverage is gated at 100%; `registration.py`, `job.py` and `sync_dataset.py`
are omitted, since they are exercised by the Code Ocean data run rather than by
local unit tests.

## Level of Support
We are not currently supporting this code, but simply releasing it to the
community AS IS but are not able to provide any guarantees of support. The
community is welcome to submit issues, but you should not expect an active
response.
